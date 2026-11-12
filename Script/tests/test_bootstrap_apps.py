"""Run shared app bootstrap with temporary repositories and stub cloud commands."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = "arn:aws:states:us-east-1:123456789012:stateMachine:test-training"


class AppBootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.apps = self.directory / "apps"
        self.calls = self.directory / "docker-calls"
        for project in ("backend", "frontend"):
            target = self.apps / project
            (target / ".git").mkdir(parents=True)
            (target / ".env.example").touch()
        self.script = self.directory / "bootstrap_apps.sh"
        self.script.write_text(
            (ROOT / "bootstrap_apps.sh").read_text().replace("/opt/fyp", str(self.apps))
        )
        shutil.copy(ROOT / "configure_training_callback.py", self.directory)
        self.stub("curl", "print('192.0.2.10')")
        self.stub("openssl", "import sys\nprint('0' * (int(sys.argv[-1]) * 2))")
        self.stub(
            "aws",
            "import json, os, sys\n"
            "if os.environ.get('TEST_CALLBACK_FAILURE'):\n"
            "    sys.exit(254)\n"
            "print(json.dumps({'Parameter': {'Value': 'fixture-callback'}}))",
        )
        self.stub(
            "docker",
            "import os, sys\nfrom pathlib import Path\n"
            "with Path(os.environ['TEST_DOCKER_CALLS']).open('a') as log:\n"
            "    log.write(' '.join(sys.argv[1:]) + '\\n')",
        )
        self.environment = {
            **os.environ,
            "PATH": str(self.directory) + os.pathsep + os.environ["PATH"],
            "TRAINING_STATE_MACHINE_ARN": WORKFLOW,
            "TRAINING_CALLBACK_TOKEN_PARAMETER": "/test/callback",
            "AWS_REGION": "us-east-1",
            "TEST_DOCKER_CALLS": str(self.calls),
            "TEST_CALLBACK_FAILURE": "",
        }

    def stub(self, name, code):
        path = self.directory / name
        path.write_text(f"#!{sys.executable}\n{code}\n")
        path.chmod(0o755)

    def run_bootstrap(self, **overrides):
        return subprocess.run(
            ["bash", str(self.script)],
            env={**self.environment, **overrides},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_writes_workflow_and_callback_before_starting_both_apps(self):
        result = self.run_bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        backend = (self.apps / "backend/.env").read_text()
        self.assertIn("TRAINING_STATE_MACHINE_ARN=" + WORKFLOW, backend)
        self.assertIn('TRAINING_CALLBACK_TOKEN="fixture-callback"', backend)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn(str(self.apps / "backend"), calls[0])
        self.assertIn(str(self.apps / "frontend"), calls[1])

    def test_missing_workflow_fails_before_modifying_application_files(self):
        result = self.run_bootstrap(TRAINING_STATE_MACHINE_ARN="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("TRAINING_STATE_MACHINE_ARN", result.stderr)
        self.assertFalse((self.apps / "backend/.env").exists())
        self.assertFalse(self.calls.exists())

    def test_callback_failure_does_not_start_apps(self):
        result = self.run_bootstrap(TEST_CALLBACK_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
