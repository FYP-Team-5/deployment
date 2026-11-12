"""Run the training shell workflow against temporary storage and stub commands."""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_training.sh"
RUN_ID = "b" * 32

STUB = r"""
import json
import os
import sys
from pathlib import Path

command = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["TEST_COMMAND_LOG"], "a") as log:
    log.write(command + "\n")
if command == "df":
    print("Filesystem 1024-blocks Used Available Capacity Mounted on")
    print("/dev/test-ebs 209715200 1048576 " + os.environ["TEST_FREE_KIB"] + " 1% /")
elif command == "aws":
    Path(args[3]).write_text("test download")
elif command == "apt-get" and "install" in args:
    if "DPkg::Lock::Timeout=600" not in args:
        sys.exit("dpkg lock is held; no wait was requested")
elif command == "python" and "--prepare-only" in args:
    sys.exit(int(os.environ.get("TEST_PREPARE_EXIT", "0")))
elif command == "python" and ("train.py" in args or "train_feedback.py" in args) and "--prepare-only" not in args:
    output = Path(args[args.index("--output") + 1])
    output.mkdir()
    (output / "checkpoint").touch()
    (Path(os.environ["TMPDIR"]) / "temporary-download").touch()
    paths = {name: os.environ[name] for name in (
        "HF_HOME", "PIP_CACHE_DIR", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
        "XDG_CACHE_HOME", "TMPDIR",
    )}
    with open(os.environ["TEST_TRAIN_LOG"], "a") as log:
        log.write(json.dumps({"paths": paths, "args": args}) + "\n")
    exit_key = "TEST_FEEDBACK_EXIT" if "train_feedback.py" in args else "TEST_TRAIN_EXIT"
    sys.exit(int(os.environ.get(exit_key, "0")))
"""


class TrainingStorageTests(unittest.TestCase):
    def test_trainer_iam_allows_every_downloaded_script_and_feedback_stage_input(self):
        policy = (SCRIPT.parents[1] / "autotraining-gpu.tf").read_text()
        downloads = re.findall(
            r'aws s3 cp "s3://\$BUCKET/scripts/([^\"]+)"', SCRIPT.read_text()
        )
        self.assertIn("train_feedback.py", downloads)
        for filename in downloads:
            with self.subTest(filename=filename):
                self.assertIn(
                    f'"${{aws_s3_bucket.autotraining.arn}}/scripts/{filename}"', policy
                )
        self.assertIn(
            '"${aws_s3_bucket.autotraining.arn}/candidates/*/grading/result.json"',
            policy,
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.work = self.directory / "work"
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        (self.work / "model/.git").mkdir(parents=True)
        (self.work / ".venv/bin").mkdir(parents=True)
        (self.work / "bin").mkdir()
        for command in (
            "df",
            "flock",
            "systemd-run",
            "nvidia-smi",
            "aws",
            "gcc",
            "git",
            "curl",
            "python3",
        ):
            self.make_stub(self.bin / command)
        self.make_stub(self.work / ".venv/bin/python")
        self.make_stub(self.work / "bin/uv")
        self.command_log = self.directory / "commands.log"
        self.train_log = self.directory / "training.jsonl"
        self.environment = {
            **os.environ,
            "PATH": str(self.bin)
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "FYP_TRAINING_WORK_DIRECTORY": str(self.work),
            "FYP_TRAINING_MIN_FREE_GIB": "60",
            "TEST_COMMAND_LOG": str(self.command_log),
            "TEST_TRAIN_LOG": str(self.train_log),
            "TEST_FREE_KIB": str(100 * 1024 * 1024),
            "TEST_TRAIN_EXIT": "0",
        }

    def make_stub(self, path):
        path.write_text(f"#!{sys.executable}\n" + STUB)
        path.chmod(0o755)

    def run_training(self, **overrides):
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "example-bucket",
                "us-east-1",
                "a" * 64,
                "12345678-1234-1234-1234-123456789abc",
                RUN_ID,
            ],
            env={**self.environment, **overrides},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_all_training_paths_use_persistent_workspace_without_nvme(self):
        result = self.run_training()
        self.assertEqual(result.returncode, 0, result.stderr)
        stages = [json.loads(line) for line in self.train_log.read_text().splitlines()]
        self.assertEqual(len(stages), 2)
        self.assertIn("train.py", stages[0]["args"])
        self.assertIn("train_feedback.py", stages[1]["args"])
        training = stages[0]
        for path in training["paths"].values():
            self.assertTrue(Path(path).is_relative_to(self.work))
        for name in (
            "HF_HOME",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "UV_PYTHON_INSTALL_DIR",
            "XDG_CACHE_HOME",
        ):
            self.assertTrue(Path(training["paths"][name]).is_dir())
        self.assertFalse(Path(training["paths"]["TMPDIR"]).exists())
        self.assertFalse((self.work / f"output-{RUN_ID}").exists())
        self.assertFalse((self.work / f"feedback-output-{RUN_ID}").exists())
        self.assertTrue((self.work / f"completed-{RUN_ID}").is_file())

    def test_completed_run_keeps_cache_and_does_not_retrain_after_restart(self):
        first = self.run_training()
        self.assertEqual(first.returncode, 0, first.stderr)
        cached = self.work / "hf-cache/model-shard"
        cached.touch()
        # A redelivery should return before storage checks or downloads.
        second = self.run_training(TEST_FREE_KIB="0")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue(cached.exists())
        self.assertEqual(len(self.train_log.read_text().splitlines()), 2)

    def test_first_boot_package_install_waits_for_the_dpkg_lock(self):
        self.make_stub(self.bin / "apt-get")
        startup = self.directory / "bash-startup"
        startup.write_text(
            "command() {\n"
            '    if [[ "$*" == "-v gcc" ]]; then return 1; fi\n'
            '    builtin command "$@"\n'
            "}\n"
        )
        result = self.run_training(BASH_ENV=str(startup))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.command_log.read_text().splitlines().count("apt-get"), 2)
        self.assertTrue((self.work / f"completed-{RUN_ID}").is_file())

    def test_feedback_failure_does_not_mark_the_pair_completed(self):
        result = self.run_training(TEST_FEEDBACK_EXIT="1")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.train_log.read_text().splitlines()), 2)
        self.assertFalse((self.work / f"completed-{RUN_ID}").exists())
        self.assertFalse((self.work / f"feedback-output-{RUN_ID}").exists())

    def test_missing_feedback_targets_fail_before_either_training_stage(self):
        result = self.run_training(TEST_PREPARE_EXIT="1")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.train_log.exists())
        self.assertFalse((self.work / f"completed-{RUN_ID}").exists())

    def test_insufficient_disk_space_fails_before_downloads(self):
        result = self.run_training(TEST_FREE_KIB=str(1024 * 1024))
        self.assertEqual(result.returncode, 1)
        self.assertIn("needs at least 60 GiB free", result.stderr)
        self.assertNotIn("aws", self.command_log.read_text().splitlines())
        self.assertFalse(self.train_log.exists())

    def test_unknown_free_space_fails_before_downloads(self):
        result = self.run_training(TEST_FREE_KIB="unknown")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot determine free workspace storage", result.stderr)
        self.assertNotIn("aws", self.command_log.read_text().splitlines())

    def test_failed_training_cleans_only_run_files_and_preserves_caches(self):
        result = self.run_training(TEST_TRAIN_EXIT="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertTrue((self.work / "hf-cache").is_dir())
        self.assertTrue((self.work / ".venv/bin/python").is_file())
        self.assertFalse((self.work / f"tmp-{RUN_ID}").exists())
        self.assertFalse((self.work / f"output-{RUN_ID}").exists())
        self.assertFalse((self.work / f"completed-{RUN_ID}").exists())

    def test_free_space_budget_can_be_configured(self):
        result = self.run_training(
            FYP_TRAINING_MIN_FREE_GIB="1", TEST_FREE_KIB=str(2 * 1024 * 1024)
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_space_budget_is_rejected(self):
        for value in ("0", "-1", "1.5", "unknown"):
            with self.subTest(value=value):
                result = self.run_training(FYP_TRAINING_MIN_FREE_GIB=value)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
