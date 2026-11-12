"""Offline data, artifact and publication tests for paired training."""

import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import publish
import train_feedback

FIELDS = [
    "id",
    "question",
    "desired_answer",
    "student_answer",
    "criteria[]",
    "criteria_score[]",
    "criteria_max_score[]",
    "student_score",
    "feedback",
]
RUN_ID = "b" * 32


def dataset(tmp_path, count=10):
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "id": str(index),
                    "question": f"Question {index}",
                    "desired_answer": "Reference",
                    "student_answer": "Student answer",
                    "criteria[]": '["First", "Second"]',
                    "criteria_score[]": "[2, 0]",
                    "criteria_max_score[]": "[2, 3]",
                    "student_score": 2,
                    "feedback": f"Feedback target {index}",
                }
            )
    return path


def args_for(path):
    return SimpleNamespace(
        dataset=str(path),
        checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        run_id=RUN_ID,
        version_id="12345678-1234-1234-1234-123456789abc",
        bucket="bucket",
        output=str(path.parent / "feedback-output"),
        repo="org/grading",
        feedback_repo="org/feedback",
        hf_token_parameter="/hf",
        callback_token_parameter="/callback",
        callback_url="http://private/api",
    )


def grading_manifest(args):
    return {
        "run_id": args.run_id,
        "dataset_checksum": args.checksum,
        "dataset_version": args.version_id,
        "status": "completed",
        "completed_at": "2026-11-11T00:00:00+00:00",
        "base_model": "Qwen/Qwen3.5-9B",
        "metrics": {"valid_examples": 1, "final_score_mae": 0},
        "baseline_metrics": {},
        "configuration": {
            "base_model_revision": "d" * 40,
            "split_strategy": "question_group_80_10_10",
        },
        "dataset_stats": {"training_rows": 10},
        "feedback_required": True,
    }


def test_feedback_split_matches_grading_question_groups_and_uses_only_targets(tmp_path):
    path = dataset(tmp_path)
    examples, stats = train_feedback.prepare_examples(path)
    shuffled = list(range(10))
    random.Random(42).shuffle(shuffled)
    assert stats["split_rows"] == {"train": 8, "validation": 1, "test": 1}
    for name, indices in (
        ("train", shuffled[:8]),
        ("validation", shuffled[8:9]),
        ("test", shuffled[9:]),
    ):
        assert {row["row_index"] for row in examples[name]} == set(indices)
    example = examples["train"][0]
    assert "[MET] First" in example["prompt"][1]["content"]
    assert "[NOT MET] Second" in example["prompt"][1]["content"]
    assert "Feedback target" not in example["prompt"][1]["content"]
    assert example["completion"][0]["content"].startswith("Feedback target")


def test_missing_validation_targets_fail_instead_of_moving_test_questions(tmp_path):
    path = dataset(tmp_path)
    order = list(range(10))
    random.Random(42).shuffle(order)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[order[8]]["feedback"] = ""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="validation question splits"):
        train_feedback.prepare_examples(path)


def test_preflight_uses_no_gpu_or_aws(tmp_path):
    args = args_for(dataset(tmp_path))
    args.prepare_only = True
    with (
        patch.object(train_feedback, "fit") as fit,
        patch.object(train_feedback.boto3, "client") as aws,
    ):
        train_feedback.run(args)
    fit.assert_not_called()
    aws.assert_not_called()


@pytest.mark.parametrize(
    "failure", [None, "missing_weights", "training_failed", "wrong_run"]
)
def test_completion_manifest_requires_both_stages_and_feedback_artifacts(
    tmp_path, failure
):
    args = args_for(dataset(tmp_path))
    grading = grading_manifest(args)
    if failure == "wrong_run":
        grading["run_id"] = "f" * 32
    aws = MagicMock()
    aws.get_object.return_value = {"Body": io.BytesIO(json.dumps(grading).encode())}

    def fit(examples, output, revision):
        assert revision == grading["configuration"]["base_model_revision"]
        assert len(examples["train"]) == 8
        if failure == "training_failed":
            raise RuntimeError("Training failed")
        (output / "adapter_config.json").write_text("config")
        if failure != "missing_weights":
            (output / "adapter_model.safetensors").write_bytes(b"weights")
        return {"LORA_R": 16, "LORA_ALPHA": 32, "base_model_revision": revision}

    with (
        patch.object(train_feedback.boto3, "client", return_value=aws),
        patch.object(train_feedback, "fit", side_effect=fit) as trainer,
    ):
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                train_feedback.run(args)
            aws.put_object.assert_not_called()
            if failure == "wrong_run":
                trainer.assert_not_called()
            return
        train_feedback.run(args)
    manifest = json.loads(aws.put_object.call_args.kwargs["Body"])
    assert aws.method_calls[-1][0] == "put_object"
    assert aws.upload_file.call_count == 3
    assert manifest["metrics"] == grading["metrics"]
    assert "metrics" not in manifest["feedback"]
    assert manifest["feedback"]["configuration"]["LORA_R"] == 16
    assert manifest["feedback"]["configuration"]["held_out_evaluation"] is False
    assert aws.put_object.call_args.kwargs["Key"].endswith(f"/{RUN_ID}/result.json")


def test_paired_publication_waits_for_both_branches_then_sends_one_callback(tmp_path):
    args = args_for(dataset(tmp_path))
    manifest = grading_manifest(args)
    manifest["feedback"] = {
        "configuration": {"LORA_R": 16},
        "dataset_stats": {"split_rows": {"train": 8}},
    }
    aws = MagicMock()
    aws.get_parameter.return_value = {"Parameter": {"Value": "secret"}}
    with (
        patch.object(publish.boto3, "client", return_value=aws),
        patch.object(publish, "get_json", side_effect=[None, manifest]),
        patch.object(
            publish, "publish_adapter", side_effect=["c" * 40, "e" * 40]
        ) as upload,
        patch.object(publish, "put_json") as save,
        patch.object(publish, "notify") as notify,
    ):
        publish.publish(args)
    assert [call.args[3] for call in upload.call_args_list] == [
        "org/grading",
        "org/feedback",
    ]
    assert [call.args[-1] for call in upload.call_args_list] == ["grading", "feedback"]
    candidate = notify.call_args.args[-1]
    assert candidate["hf_revision"] == "c" * 40
    assert candidate["feedback"]["hf_revision"] == "e" * 40
    assert candidate["feedback"]["hf_branch"] == "candidate-" + RUN_ID
    assert "metrics" not in candidate["feedback"]
    assert len(save.call_args_list) == 2
    assert save.call_args_list[0].args[-1]["notified"] is False
    assert save.call_args_list[-1].args[-1]["notified"] is True


def test_interrupted_second_upload_cannot_register_partial_pair(tmp_path):
    args = args_for(dataset(tmp_path))
    manifest = grading_manifest(args)
    manifest["feedback"] = {"configuration": {"LORA_R": 16}, "dataset_stats": {}}
    aws = MagicMock()
    aws.get_parameter.return_value = {"Parameter": {"Value": "secret"}}
    with (
        patch.object(publish.boto3, "client", return_value=aws),
        patch.object(publish, "get_json", side_effect=[None, manifest]),
        patch.object(
            publish,
            "publish_adapter",
            side_effect=["c" * 40, RuntimeError("Feedback HF unavailable")],
        ),
        patch.object(publish, "put_json") as save,
        patch.object(publish, "notify") as notify,
        pytest.raises(RuntimeError),
    ):
        publish.publish(args)
    save.assert_not_called()
    notify.assert_not_called()


def test_retry_reuses_grading_commit_after_feedback_upload_interruption(tmp_path):
    args = args_for(dataset(tmp_path))
    manifest = grading_manifest(args)
    manifest["feedback"] = {"configuration": {"LORA_R": 16}, "dataset_stats": {}}
    remote = {}
    attempts = []
    branches = []

    class MissingEntry(Exception):
        pass

    class Api:
        def __init__(self, token):
            pass

        def create_branch(self, **kwargs):
            branches.append(kwargs["branch"])

        def model_info(self, repo, revision):
            return SimpleNamespace(sha=remote[repo]["revision"])

        def upload_folder(self, **kwargs):
            repo = kwargs["repo_id"]
            attempts.append(repo)
            if repo == args.feedback_repo and attempts.count(repo) == 1:
                raise RuntimeError("Feedback upload interrupted")
            revision = ("c" if repo == args.repo else "e") * 40
            metadata = json.loads(
                (Path(kwargs["folder_path"]) / "candidate.json").read_text()
            )
            remote[repo] = {"metadata": metadata, "revision": revision}
            return SimpleNamespace(oid=revision)

    def download(repo_id, **kwargs):
        if repo_id not in remote:
            raise MissingEntry()
        target = tmp_path / (repo_id.replace("/", "-") + ".json")
        target.write_text(json.dumps(remote[repo_id]["metadata"]))
        return str(target)

    aws = MagicMock()
    aws.get_parameter.return_value = {"Parameter": {"Value": "secret"}}
    aws.download_file.side_effect = lambda bucket, key, target: Path(
        target
    ).write_bytes(b"artifact")
    modules = {
        "huggingface_hub": SimpleNamespace(HfApi=Api, hf_hub_download=download),
        "huggingface_hub.errors": SimpleNamespace(EntryNotFoundError=MissingEntry),
    }
    with (
        patch.dict(sys.modules, modules),
        patch.object(publish.boto3, "client", return_value=aws),
        patch.object(publish, "get_json", side_effect=[None, manifest, None, manifest]),
        patch.object(publish, "put_json"),
        patch.object(publish, "notify") as notify,
    ):
        with pytest.raises(RuntimeError, match="Feedback upload interrupted"):
            publish.publish(args)
        notify.assert_not_called()
        publish.publish(args)
        notify.assert_called_once()
    assert attempts == [args.repo, args.feedback_repo, args.feedback_repo]
    assert all(branch == "candidate-" + RUN_ID for branch in branches)
    assert "metrics" not in remote[args.feedback_repo]["metadata"]
