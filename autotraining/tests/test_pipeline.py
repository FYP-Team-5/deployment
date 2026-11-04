"""Offline regression tests; no AWS credentials or GPU are used."""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.update({
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "STATE_TABLE": "unit-tests",
    "ARTIFACT_BUCKET": "example-bucket",
    "BACKEND_INSTANCE_ID": "i-00000000000000001",
    "TRAINING_INSTANCE_ID": "i-00000000000000002",
    "TRAIN_LOG_GROUP": "/fyp/test/train",
})
sys.path.insert(0, str(ROOT))
import coordinator
import execution
import train
import transfer


class DatasetTests(unittest.TestCase):
    def dataset(self, directory, partial=False):
        source = Path(directory) / "dataset.csv"
        fields = ["id", "question", "desired_answer", "student_answer",
                  "criteria[]", "criteria_score[]", "criteria_max_score[]",
                  "student_score"]
        with source.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for i in range(12):
                is_partial = partial and i == 11
                writer.writerow({
                    "id": str(i),
                    "question": f"Question {i}",
                    "desired_answer": "Known expected response",
                    "student_answer": "Candidate answer",
                    "criteria[]": json.dumps(["first", "second"]),
                    "criteria_score[]": json.dumps([1.0 if is_partial else 2.0, 0.0]),
                    "criteria_max_score[]": json.dumps([2.0, 1.0]),
                    "student_score": 1.0 if is_partial else 2.0,
                })
        return source

    def test_normalizes_full_credit_float_and_reports_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.dataset(folder, partial=True)
            processed, stats = train.normalize_dataset(source)
            self.assertEqual(stats["source_rows"], 12)
            self.assertEqual(stats["training_rows"], 11)
            self.assertEqual(stats["skipped_partial_credit"], 1)
            with processed.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(json.loads(rows[0]["criteria_score[]"]), [1, 0])

    def test_original_checksum_is_not_the_normalized_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.dataset(folder)
            normal, _ = train.normalize_dataset(source)
            self.assertNotEqual(source.read_bytes(), normal.read_bytes())


class CoordinatorTests(unittest.TestCase):
    def test_unchanged_checksum_skips_gpu(self):
        job = {"checksum": "a" * 64, "version_id": "f" * 36, "row_count": 12}
        class DB:
            def get_item(self, **kwargs):
                return {"Item": {"checksum": "a" * 64}}
        with patch.object(transfer, "snapshot", return_value=job), patch.object(transfer, "DB", DB()):
            self.assertFalse(transfer.prepare(object())["train"])

    def test_new_checksum_acquires_lock(self):
        job = {"checksum": "b" * 64, "version_id": "f" * 36, "row_count": 12}
        class DB:
            def __init__(self):
                self.stored = None
            def get_item(self, **kwargs):
                return {"Item": {"checksum": "a" * 64}}
            def put_item(self, **kwargs):
                self.stored = kwargs
        db = DB()
        with patch.object(transfer, "snapshot", return_value=job), patch.object(transfer, "DB", db):
            self.assertTrue(transfer.prepare(object())["train"])
        self.assertEqual(db.stored["Item"]["checksum"], "b" * 64)

    def test_ssm_readiness_has_bounded_retry_counter(self):
        class Offline:
            def describe_instance_information(self, **kwargs):
                return {"InstanceInformationList": []}
        with patch.object(execution, "SSM", Offline()):
            self.assertEqual(execution.launch({"checksum": "a" * 64}, 4),
                             {"ready": False, "attempt": 5})

    def test_poll_reports_success_and_failure(self):
        with patch.object(execution, "read_command", return_value={"Status": "Success"}):
            self.assertTrue(execution.poll("dummy")["done"])
        with patch.object(execution, "read_command", return_value={"Status": "Failed"}):
            with self.assertRaises(RuntimeError):
                execution.poll("dummy")

    def test_invalid_candidate_cannot_become_latest(self):
        class InvalidS3:
            def get_object(self, **kwargs):
                return {"Body": io.BytesIO(b'{"status":"incomplete"}')}
        class DB:
            def put_item(self, **kwargs):
                raise AssertionError("Invalid candidate was accepted")
        with patch.object(execution, "S3", InvalidS3()), patch.object(execution, "DB", DB()):
            with self.assertRaises(ValueError):
                execution.finish({"checksum": "a" * 64, "version_id": "f" * 36})

    def test_failed_stop_keeps_dataset_lock(self):
        with patch.object(execution, "stop", side_effect=RuntimeError("EC2 API unavailable")), \
             patch.object(execution, "unlock") as release:
            with self.assertRaises(RuntimeError):
                execution.failure({"checksum": "a" * 64})
            release.assert_not_called()

    def test_stop_releases_lock_after_ec2_stop(self):
        trace = []
        with patch.object(execution, "stop", side_effect=lambda: trace.append("stop") or {}), \
             patch.object(transfer, "unlock", side_effect=lambda _: trace.append("unlock")):
            coordinator.handler({"action": "stop", "job": {"checksum": "a" * 64}}, None)
        self.assertEqual(trace, ["stop", "unlock"])


class WorkflowTests(unittest.TestCase):
    def test_valid_state_graph(self):
        machine = json.loads((ROOT / "workflow.asl.json.tftpl").read_text())
        states = machine["States"]
        self.assertIn(machine["StartAt"], states)
        for name, state in states.items():
            targets = [state.get("Next"), state.get("Default")]
            targets += [edge.get("Next") for edge in state.get("Choices", [])]
            targets += [edge.get("Next") for edge in state.get("Catch", [])]
            for target in filter(None, targets):
                self.assertIn(target, states, (name, target))
        self.assertEqual(states["Prepare"]["Parameters"]["action"], "prepare")
        self.assertEqual(states["Complete"]["Parameters"]["action"], "complete")
        self.assertEqual(states["FailAndStop"]["Parameters"]["action"], "fail")


if __name__ == "__main__":
    unittest.main()
