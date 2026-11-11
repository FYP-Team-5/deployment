"""Offline regression tests; no AWS credentials or GPU are used."""

import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(
    {
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "ARTIFACT_BUCKET": "example-bucket",
        "BACKEND_INSTANCE_ID": "i-backend",
        "TRAINING_INSTANCE_ID": "i-trainer",
        "TRAIN_LOG_GROUP": "/fyp/test/train",
    }
)
sys.path.insert(0, str(ROOT))
import coordinator
import execution
import publish
import state
import train
import transfer

FIELDS = [
    "id",
    "question",
    "desired_answer",
    "student_answer",
    "criteria[]",
    "criteria_score[]",
    "criteria_max_score[]",
    "student_score",
]
JOB = {
    "checksum": "a" * 64,
    "version_id": "12345678-1234-1234-1234-123456789abc",
    "row_count": 12,
    "run_id": "b" * 32,
    "execution_id": "arn:execution:test",
    "started_at": 100,
}


class MemoryState:
    def __init__(self):
        self.items = {}

    def tag(self, value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def get_object(self, Bucket, Key):
        key = Key.removeprefix("control/").removesuffix(".json")
        if key not in self.items:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {
            "Body": io.BytesIO(json.dumps(self.items[key]).encode()),
            "ETag": self.tag(self.items[key]),
        }

    def put_object(self, Bucket, Key, Body, **kwargs):
        key = Key.removeprefix("control/").removesuffix(".json")
        current = self.items.get(key)
        if kwargs.get("IfNoneMatch") == "*" and current is not None:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if kwargs.get("IfMatch") and (
            current is None or self.tag(current) != kwargs["IfMatch"]
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.items[key] = json.loads(Body)


class DatasetTests(unittest.TestCase):
    def dataset(self, directory, change=None, count=12):
        source = Path(directory) / "dataset.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            for i in range(count):
                row = {
                    "id": str(i),
                    "question": f"Question {i},\nnext line",
                    "desired_answer": "Expected response",
                    "student_answer": "",
                    "criteria[]": '["first, with comma", "second"]',
                    "criteria_score[]": "[2.0, 0.0]",
                    "criteria_max_score[]": "[2.0, 1.0]",
                    "student_score": "2.0",
                }
                if i == count - 1 and change:
                    row.update(change)
                writer.writerow(row)
        return source

    def test_normalizes_weighted_labels_preserving_maxima_and_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            processed, stats = train.normalize_dataset(self.dataset(folder))
            self.assertEqual(stats["training_rows"], 12)
            with processed.open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["criteria_score[]"]), [1, 0])
            self.assertEqual(json.loads(row["criteria_max_score[]"]), [2, 1])
            self.assertEqual(float(row["student_score"]), 2)
            self.assertEqual(row["question"], "Question 0,\nnext line")

    def test_partial_credit_is_counted_and_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            _, stats = train.normalize_dataset(
                self.dataset(
                    folder, {"criteria_score[]": "[1, 0]", "student_score": "1"}
                )
            )
            self.assertEqual(stats["training_rows"], 11)
            self.assertEqual(stats["skipped_partial_credit"], 1)

    def test_blank_desired_answer_is_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            _, stats = train.normalize_dataset(
                self.dataset(folder, {"desired_answer": ""})
            )
            self.assertEqual(stats["skipped_missing_answer"], 1)

    def test_rejects_invalid_arrays_numbers_and_totals(self):
        for change in (
            {"criteria[]": '["one"]'},
            {"criteria[]": '"text"'},
            {"criteria[]": '["", "two"]'},
            {"criteria_score[]": "[true, 0]"},
            {"criteria_score[]": "[3, 0]"},
            {"criteria_score[]": "[-1, 0]"},
            {"criteria_max_score[]": "[2, 0]"},
            {"criteria_max_score[]": "[NaN, 1]"},
            {"criteria_score[]": "[Infinity, 0]"},
            {"student_score": "nan"},
            {"student_score": "1"},
            {"question": ""},
            {"id": ""},
        ):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as folder,
                self.assertRaises(ValueError),
            ):
                train.normalize_dataset(self.dataset(folder, change))

    def test_accepts_one_unique_question(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.dataset(folder)
            with source.open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["question"] = "same question"
            with source.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            _, stats = train.normalize_dataset(source)
            self.assertEqual(stats["unique_questions"], 1)
            self.assertEqual(stats["training_rows"], 12)

    def test_rejects_no_compatible_rows(self):
        for count, change in ((0, None), (1, {"desired_answer": ""})):
            with (
                self.subTest(count=count),
                tempfile.TemporaryDirectory() as folder,
                self.assertRaisesRegex(ValueError, "No compatible training rows"),
            ):
                train.normalize_dataset(self.dataset(folder, change, count=count))


class RunnerTests(unittest.TestCase):
    dataset = DatasetTests.dataset

    def test_manifest_published_last_with_effective_configuration(self):
        for row_count in (1, 2, 9, 10, 12):
            with self.subTest(row_count=row_count):
                self.check_manifest(row_count)

    def check_manifest(self, row_count):
        with tempfile.TemporaryDirectory() as folder:
            source = self.dataset(folder, count=row_count)
            notebook = Path(folder) / "reference.ipynb"
            cells = [{"cell_type": "markdown", "source": ""} for _ in range(47)]
            config_names = [
                "LORA_DROPOUT",
                "LEARNING_RATE",
                "WEIGHT_DECAY",
                "NUM_EPOCHS",
                "TRAIN_BATCH_SIZE",
                "GRADIENT_ACCUMULATION_STEPS",
                "VALIDATIONS_PER_EPOCH",
                "LR_REDUCTION_FACTOR",
                "LR_PATIENCE",
                "MIN_LEARNING_RATE",
                "EARLY_STOPPING_PATIENCE",
                "EARLY_STOPPING_THRESHOLD",
                "GENERATION_MIN_TOKENS",
                "GENERATION_TOKENS_PER_CRITERION",
            ]
            cells[6] = {
                "cell_type": "code",
                "source": [f"{name} = 1\n" for name in config_names],
            }
            cells[8] = {
                "cell_type": "code",
                "source": f"records = [0] * {row_count}\ndataset = records",
            }
            cells[16] = {
                "cell_type": "code",
                "source": (
                    "assert len(dataset) >= 10, 'Small datasets must bypass the reference split'\n"
                    "train_dataset, validation_dataset, test_dataset = dataset[:-2], dataset[-2:-1], dataset[-1:]"
                ),
            }
            cells[20] = {
                "cell_type": "code",
                "source": (
                    "if len(dataset) < 10:\n"
                    "    assert train_dataset is dataset\n"
                    "    assert validation_dataset is dataset\n"
                    "    assert test_dataset is dataset\n"
                    "else:\n"
                    "    assert len(train_dataset) + len(validation_dataset) + len(test_dataset) == len(dataset)"
                ),
            }
            cells[44] = {
                "cell_type": "code",
                "source": (
                    "from pathlib import Path\nfrom types import SimpleNamespace\n"
                    "evaluation_completed = True\n"
                    "finetuned_metrics = {'valid_examples': 2, 'final_score_mae': 0.5, 'f1': float('nan')}\n"
                    "baseline_metrics = {'valid_examples': 2, 'final_score_mae': 1.0}\n"
                    "sequence_limit = MAX_SEQ_LENGTH\n"
                    "bnb_config = SimpleNamespace(to_dict=lambda: {'load_in_4bit': True})\n"
                    "compute_dtype = 'torch.bfloat16'\ntarget_modules = ['model.layers.0.q_proj']\n"
                    "model = SimpleNamespace(config=SimpleNamespace(_commit_hash=BASE_REVISION))\n"
                    "for name in ('adapter_config.json', 'adapter_model.safetensors'):\n"
                    "    (Path(OUTPUT_DIR) / name).write_text('artifact')\n"
                ),
            }
            notebook.write_text(json.dumps({"cells": cells}))
            args = SimpleNamespace(
                dataset=str(source),
                checksum=hashlib.sha256(source.read_bytes()).hexdigest(),
                notebook=str(notebook),
                output=str(Path(folder) / "output"),
                bucket="example-bucket",
                version_id=JOB["version_id"],
                run_id=JOB["run_id"],
            )
            api_factory = MagicMock(
                return_value=SimpleNamespace(
                    model_info=lambda _: SimpleNamespace(sha="fixed-revision")
                )
            )
            hf = SimpleNamespace(HfApi=api_factory)
            s3 = MagicMock()
            with (
                patch.dict(sys.modules, {"huggingface_hub": hf}),
                patch.object(train.boto3, "client", return_value=s3),
                patch("importlib.metadata.version", return_value="test-version"),
            ):
                train.run(args)
            api_factory.assert_called_once_with(token=False)
            manifest = json.loads(s3.put_object.call_args.kwargs["Body"])
            self.assertEqual(manifest["run_id"], JOB["run_id"])
            self.assertEqual(manifest["configuration"]["LORA_R"], 48)
            self.assertEqual(manifest["configuration"]["LORA_ALPHA"], 96)
            self.assertEqual(manifest["configuration"]["MAX_SEQ_LENGTH"], 4096)
            self.assertEqual(manifest["dataset_rows"], row_count)
            self.assertEqual(manifest["dataset_stats"]["training_rows"], row_count)
            self.assertEqual(
                manifest["configuration"]["held_out_evaluation"], row_count >= 10
            )
            self.assertEqual(
                manifest["configuration"]["split_strategy"],
                "question_group_80_10_10" if row_count >= 10 else "reuse_training_data",
            )
            self.assertEqual(
                manifest["configuration"]["base_model_revision"], "fixed-revision"
            )
            self.assertIsNone(manifest["metrics"]["f1"])
            self.assertEqual(manifest["baseline_metrics"]["final_score_mae"], 1.0)
            self.assertEqual(s3.method_calls[-1][0], "put_object")
            self.assertEqual(s3.upload_file.call_count, 3)

    def test_checksum_mismatch_fails_before_training_or_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(dataset=str(self.dataset(folder)), checksum="f" * 64)
            with (
                patch.object(train.boto3, "client") as client,
                self.assertRaises(ValueError),
            ):
                train.run(args)
            client.assert_not_called()


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.db = MemoryState()
        self.patches = [patch.object(state, "S3", self.db)]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def prepare(self):
        with patch.object(transfer, "snapshot", return_value=dict(JOB)):
            return transfer.prepare(object(), JOB["execution_id"])

    def test_unchanged_latest_skips_gpu(self):
        self.db.items["success/" + JOB["checksum"]] = {"checksum": JOB["checksum"]}
        self.assertFalse(self.prepare()["train"])
        self.assertFalse(self.db.items.get("lock", {}).get("execution_id"))

    def test_manual_request_retrains_even_a_published_checksum(self):
        self.db.items["success/" + JOB["checksum"]] = {"checksum": JOB["checksum"]}
        self.db.items["trained/" + JOB["checksum"]] = dict(JOB)
        with patch.object(transfer, "snapshot", return_value=dict(JOB)):
            result = transfer.prepare(
                "command",
                JOB["execution_id"],
                {
                    "force_retrain": True,
                    "requested_by": "2000001",
                },
            )
        self.assertTrue(result["train"])
        self.assertEqual(result["requested_by"], "2000001")
        self.assertEqual(self.db.items["lock"]["execution_id"], JOB["execution_id"])

    def test_manual_request_still_cannot_overlap_gpu_owner(self):
        self.db.items["lock"] = dict(JOB, execution_id="another-execution")
        with patch.object(transfer, "snapshot", return_value=dict(JOB)):
            result = transfer.prepare(
                "command", JOB["execution_id"], {"force_retrain": True}
            )
        self.assertEqual(result["reason"], "already_running")
        self.assertEqual(self.db.items["lock"]["execution_id"], "another-execution")

    def test_unpublished_manual_rerun_is_not_masked_by_older_success(self):
        self.db.items["success/" + JOB["checksum"]] = dict(JOB, run_id="old-run")
        self.db.items["trained/" + JOB["checksum"]] = dict(JOB, run_id="new-run")
        result = self.prepare()
        self.assertFalse(result["train"])
        self.assertTrue(result["publish"])
        self.assertEqual(result["run_id"], "new-run")
        self.assertNotIn("lock", self.db.items)

    def test_manual_request_replay_reuses_the_same_run(self):
        with patch.object(transfer, "snapshot", return_value=dict(JOB)):
            first = transfer.prepare(
                "command", JOB["execution_id"], {"force_retrain": True}
            )
            replay = transfer.prepare(
                "command", JOB["execution_id"], {"force_retrain": True}
            )
        self.assertEqual(first["run_id"], replay["run_id"])

    def test_force_flag_must_be_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            transfer.prepare("command", JOB["execution_id"], {"force_retrain": "false"})

    def test_previously_trained_checksum_skips_even_if_latest_changed(self):
        self.db.items["success/" + JOB["checksum"]] = {"checksum": JOB["checksum"]}
        self.db.items["latest"] = {"checksum": "c" * 64}
        self.assertEqual(self.prepare()["reason"], "unchanged")

    def test_no_ready_dataset_skips(self):
        with patch.object(
            transfer, "snapshot", return_value={"reason": "no_ready_dataset"}
        ):
            self.assertFalse(transfer.prepare(object(), "arn:test")["train"])
        self.assertFalse(self.db.items)

    def test_new_checksum_claims_lock_and_records_run(self):
        job = self.prepare()
        self.assertTrue(job["train"])
        self.assertEqual(self.db.items["lock"]["execution_id"], JOB["execution_id"])
        self.assertNotIn("expires_at", self.db.items["lock"])
        self.assertEqual(self.db.items["runs/" + job["run_id"]]["status"], "running")

    def test_overlap_is_skipped_and_cannot_release_another_owner(self):
        self.db.items["lock"] = dict(JOB, execution_id="other")
        self.assertEqual(self.prepare()["reason"], "already_running")
        transfer.unlock(JOB)
        self.assertEqual(self.db.items["lock"]["execution_id"], "other")

    def test_prepare_is_idempotent_for_lock_owner(self):
        first = self.prepare()
        self.assertEqual(self.prepare()["run_id"], first["run_id"])

    def test_ssm_readiness_has_bounded_counter(self):
        self.db.items["lock"] = dict(JOB)
        with patch.object(execution, "SSM") as ssm:
            ssm.describe_instance_information.return_value = {
                "InstanceInformationList": []
            }
            self.assertEqual(execution.launch(JOB, 4), {"ready": False, "attempt": 5})

    def test_stale_execution_cannot_start_gpu(self):
        self.db.items["lock"] = dict(JOB, execution_id="other")
        with patch.object(execution, "EC2") as ec2, self.assertRaises(RuntimeError):
            execution.start(JOB)
        ec2.start_instances.assert_not_called()

    def test_poll_success_failure_and_pending(self):
        for status, expected in (
            ("Success", True),
            ("InProgress", False),
            ("Pending", False),
        ):
            with patch.object(
                execution, "read_command", return_value={"Status": status}
            ):
                self.assertEqual(execution.poll("dummy")["done"], expected)
        with (
            patch.object(execution, "read_command", return_value={"Status": "Failed"}),
            self.assertRaises(RuntimeError),
        ):
            execution.poll("dummy")

    def test_failed_stop_keeps_lock_for_watchdog(self):
        self.db.items["lock"] = dict(JOB)
        with (
            patch.object(execution, "stop", side_effect=RuntimeError("unavailable")),
            self.assertRaises(RuntimeError),
        ):
            execution.cleanup(JOB["execution_id"])
        self.assertIn("lock", self.db.items)

    def test_stop_waits_in_workflow_without_billing_lambda_for_a_waiter(self):
        self.db.items["lock"] = dict(JOB)
        with patch.object(execution, "EC2") as ec2:
            ec2.describe_instances.return_value = {
                "Reservations": [{"Instances": [{"State": {"Name": "running"}}]}]
            }
            self.assertFalse(execution.cleanup(JOB["execution_id"])["stopped"])
            ec2.stop_instances.assert_called_once()
            self.assertEqual(self.db.items["lock"]["execution_id"], JOB["execution_id"])
            ec2.describe_instances.return_value = {
                "Reservations": [{"Instances": [{"State": {"Name": "stopped"}}]}]
            }
            self.assertTrue(execution.cleanup(JOB["execution_id"])["stopped"])
            ec2.get_waiter.assert_not_called()
        self.assertFalse(self.db.items["lock"].get("execution_id"))

    def test_late_abort_event_cannot_stop_next_run(self):
        self.db.items["lock"] = dict(JOB, execution_id="next_execution")
        with patch.object(execution, "stop") as stop:
            coordinator.handler(
                {
                    "source": "aws.states",
                    "detail": {
                        "executionArn": JOB["execution_id"],
                        "status": "ABORTED",
                    },
                },
                None,
            )
            stop.assert_not_called()

    def test_concurrent_cleanup_is_serialized(self):
        self.db.items["lock"] = dict(JOB)

        def second_cleanup():
            with self.assertRaises(state.StateConflict):
                execution.cleanup(JOB["execution_id"])
            return {"stopped": True}

        with patch.object(execution, "stop", side_effect=second_cleanup) as stop:
            execution.cleanup(JOB["execution_id"])
        self.assertEqual(stop.call_count, 1)
        self.assertFalse(self.db.items.get("lock", {}).get("execution_id"))

    def test_expired_cleanup_lease_can_be_recovered(self):
        self.db.items["lock"] = dict(JOB, cleanup_token="expired", cleanup_until=1)
        with patch.object(execution, "stop", return_value={"stopped": True}):
            execution.cleanup(JOB["execution_id"])
        self.assertFalse(self.db.items.get("lock", {}).get("execution_id"))

    def test_watchdog_cleans_overdue_owner(self):
        self.db.items["lock"] = dict(JOB)
        with patch.object(execution, "stop", return_value={"stopped": True}):
            execution.watchdog()
        self.assertFalse(self.db.items.get("lock", {}).get("execution_id"))

    def test_watchdog_keeps_live_execution(self):
        import time

        self.db.items["lock"] = dict(JOB, started_at=int(time.time()))
        with (
            patch.object(execution.boto3, "client") as client,
            patch.object(execution, "stop") as stop,
        ):
            client.return_value.describe_execution.return_value = {"status": "RUNNING"}
            self.assertEqual(execution.watchdog(), {"running": True})
            stop.assert_not_called()

    def test_invalid_candidate_cannot_become_latest(self):
        self.db.items["lock"] = dict(JOB)
        with patch.object(execution, "S3") as s3:
            s3.get_object.return_value = {
                "Body": io.BytesIO(b'{"status":"incomplete"}')
            }
            with self.assertRaises(ValueError):
                execution.finish(JOB)
        self.assertNotIn("latest", self.db.items)

    def test_valid_candidate_records_success_and_cleanup_completes(self):
        self.db.items["lock"] = dict(JOB)
        prefix = f"candidates/{JOB['checksum']}/{JOB['run_id']}"
        manifest = {
            "status": "completed",
            "dataset_checksum": JOB["checksum"],
            "dataset_version": JOB["version_id"],
            "run_id": JOB["run_id"],
            "configuration": {"LORA_R": 48},
            "metrics": {"valid_examples": 2, "final_score_mae": 0.5},
            "adapter_uri": f"s3://example-bucket/{prefix}/adapter/",
        }
        with patch.object(execution, "S3") as s3:
            s3.get_object.return_value = {
                "Body": io.BytesIO(json.dumps(manifest).encode())
            }
            execution.finish(JOB)
            self.assertEqual(s3.head_object.call_count, 2)
        self.assertIn("trained/" + JOB["checksum"], self.db.items)
        with patch.object(execution, "stop", return_value={"stopped": True}):
            execution.cleanup(JOB["execution_id"])
        self.assertEqual(self.db.items["runs/" + JOB["run_id"]]["status"], "completed")


class S3StateTests(unittest.TestCase):
    def setUp(self):
        self.storage = MemoryState()
        self.patch = patch.object(state, "S3", self.storage)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_compare_and_swap_rejects_stale_owner(self):
        self.assertTrue(state.claim(JOB))
        value, etag = state.read("lock")
        state.claim_cleanup(JOB)
        with self.assertRaises(state.StateConflict):
            state.write("lock", value, etag=etag)

    def test_idle_generation_changes_to_prevent_aba(self):
        state.claim(JOB)
        state.unlock(JOB)
        first_idle, etag = state.read("lock")
        state.claim(dict(JOB, execution_id="second"))
        state.unlock(dict(JOB, execution_id="second"))
        self.assertNotEqual(first_idle, state.get_item("lock"))
        with self.assertRaises(state.StateConflict):
            state.write("lock", JOB, etag=etag)

    def test_absent_record_does_not_hide_permission_failures(self):
        with (
            patch.object(
                state.S3,
                "get_object",
                side_effect=ClientError(
                    {"Error": {"Code": "AccessDenied"}}, "GetObject"
                ),
            ),
            self.assertRaises(ClientError),
        ):
            state.get_item("lock")


class PublicationTests(unittest.TestCase):
    def test_callback_replay_reuses_pinned_hf_upload(self):
        receipt = {
            "run_id": JOB["run_id"],
            "hf_repo": "org/repo",
            "hf_revision": "c" * 40,
            "candidate": {"run_id": JOB["run_id"]},
            "notified": False,
        }
        args = SimpleNamespace(
            run_id=JOB["run_id"],
            checksum=JOB["checksum"],
            bucket="bucket",
            hf_token_parameter="/hf",
            callback_token_parameter="/callback",
            callback_url="http://private/api",
            repo="org/repo",
        )
        aws = MagicMock()
        aws.get_parameter.return_value = {"Parameter": {"Value": "secret"}}
        with (
            patch.object(publish.boto3, "client", return_value=aws),
            patch.object(publish, "get_json", return_value=receipt),
            patch.object(publish, "notify") as notify,
            patch.object(publish, "put_json") as save,
        ):
            publish.publish(args)
        notify.assert_called_once_with(
            args.callback_url, "secret", receipt["candidate"]
        )
        self.assertTrue(save.call_args.args[-1]["notified"])
        aws.get_parameter.assert_called_once_with(Name="/callback", WithDecryption=True)

    def test_fresh_publication_writes_candidate_branch_then_saves_receipt_before_callback(
        self,
    ):
        args = SimpleNamespace(
            run_id=JOB["run_id"],
            checksum=JOB["checksum"],
            bucket="bucket",
            hf_token_parameter="/hf",
            callback_token_parameter="/callback",
            callback_url="http://private/api",
            repo="org/repo",
        )
        manifest = {
            "run_id": JOB["run_id"],
            "dataset_checksum": JOB["checksum"],
            "dataset_version": JOB["version_id"],
            "completed_at": "2026-11-07T00:00:00+00:00",
            "base_model": "Qwen/Qwen3.5-9B",
            "metrics": {"valid_examples": 2, "final_score_mae": 0.1},
            "baseline_metrics": {},
            "configuration": {},
            "dataset_stats": {},
        }
        aws, hf = MagicMock(), MagicMock()
        aws.get_parameter.return_value = {"Parameter": {"Value": "secret"}}
        aws.download_file.side_effect = lambda bucket, key, path: Path(path).write_text(
            "artifact"
        )
        hf.upload_folder.return_value = SimpleNamespace(oid="c" * 40)

        class MissingEntry(Exception):
            pass

        def missing(**kwargs):
            raise MissingEntry()

        api_factory = MagicMock(return_value=hf)
        public_download = MagicMock(side_effect=missing)
        modules = {
            "huggingface_hub": SimpleNamespace(
                HfApi=api_factory, hf_hub_download=public_download
            ),
            "huggingface_hub.errors": SimpleNamespace(EntryNotFoundError=MissingEntry),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(publish.boto3, "client", return_value=aws),
            patch.object(publish, "get_json", side_effect=[None, manifest]),
            patch.object(publish, "put_json") as save,
            patch.object(
                publish, "notify", side_effect=RuntimeError("backend unavailable")
            ),
            self.assertRaises(RuntimeError),
        ):
            publish.publish(args)
        branch = "candidate-" + JOB["run_id"]
        self.assertEqual(hf.create_branch.call_args.kwargs["branch"], branch)
        self.assertEqual(hf.upload_folder.call_args.kwargs["revision"], branch)
        self.assertIs(public_download.call_args.kwargs["token"], False)
        self.assertEqual(
            [call.kwargs["token"] for call in api_factory.call_args_list],
            ["secret", False],
        )
        saved = save.call_args.args[-1]
        self.assertFalse(saved["notified"])
        self.assertEqual(saved["candidate"]["hf_revision"], "c" * 40)
        self.assertEqual(saved["candidate"]["output_contract"], "criteria_array_v1")

    def test_completed_training_retries_publication_without_gpu(self):
        storage = MemoryState()
        storage.items["trained/" + JOB["checksum"]] = dict(JOB)
        with (
            patch.object(state, "S3", storage),
            patch.object(transfer, "snapshot", return_value=dict(JOB)),
        ):
            result = transfer.prepare("ssm-command", "retry-execution")
        self.assertFalse(result["train"])
        self.assertTrue(result["publish"])
        self.assertFalse(storage.items.get("lock"))


class WorkflowTests(unittest.TestCase):
    def test_state_graph_and_cleanup_paths(self):
        machine = json.loads((ROOT / "workflow.asl.json.tftpl").read_text())
        states = machine["States"]
        self.assertIn(machine["StartAt"], states)
        for name, node in states.items():
            targets = [node.get("Next"), node.get("Default")]
            targets += [edge.get("Next") for edge in node.get("Choices", [])]
            targets += [edge.get("Next") for edge in node.get("Catch", [])]
            for target in filter(None, targets):
                self.assertIn(target, states, (name, target))
            if node["Type"] == "Task" and name != "FailAndStop":
                self.assertEqual(node["Catch"][0]["Next"], "FailAndStop")
                self.assertEqual(node["Catch"][0]["ResultPath"], "$.error")
        self.assertEqual(
            states["Initialize"]["Parameters"]["execution_id.$"], "$$.Execution.Id"
        )
        self.assertIn("execution_id.$", states["NextAttempt"]["Parameters"])
        self.assertEqual(states["Initialize"]["Parameters"]["request.$"], "$")
        for name in ("Prepare", "NextExportAttempt", "NextAttempt"):
            self.assertEqual(states[name]["Parameters"]["request.$"], "$.request")


if __name__ == "__main__":
    unittest.main()
