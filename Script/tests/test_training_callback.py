"""Verify callback provisioning without reading real credentials or contacting AWS."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "configure_training_callback.py"
SPEC = importlib.util.spec_from_file_location("configure_training_callback", SCRIPT)
callback = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(callback)


class CallbackConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.env_file = self.directory / ".env"
        self.original = "API_KEY=keep-this\nTRAINING_CALLBACK_TOKEN=stale\n"
        self.env_file.write_text(self.original)

    def configure(self, value):
        response = SimpleNamespace(
            returncode=0, stdout=json.dumps({"Parameter": {"Value": value}}), stderr=""
        )
        with patch.object(callback.subprocess, "run", return_value=response) as run:
            callback.configure(self.env_file, "/custom/callback", "ap-southeast-1")
        return run.call_args

    def test_uses_configured_parameter_and_keeps_credential_private(self):
        token = "fixture-callback-secret"
        call = self.configure(token)
        content = self.env_file.read_text()
        self.assertIn("API_KEY=keep-this", content)
        self.assertIn('TRAINING_CALLBACK_TOKEN="fixture-callback-secret"', content)
        self.assertNotIn("stale", content)
        self.assertEqual(self.env_file.stat().st_mode & 0o777, 0o600)
        self.assertIn("--with-decryption", call.args[0])
        self.assertIn("/custom/callback", call.args[0])
        self.assertIn("ap-southeast-1", call.args[0])
        self.assertNotIn(token, " ".join(call.args[0]))

    def test_repeated_configuration_replaces_duplicates(self):
        self.env_file.write_text(self.original + "TRAINING_CALLBACK_TOKEN=duplicate\n")
        self.configure("first")
        self.configure("second")
        content = self.env_file.read_text()
        self.assertEqual(content.count("TRAINING_CALLBACK_TOKEN="), 1)
        self.assertIn('TRAINING_CALLBACK_TOKEN="second"', content)

    def test_missing_ssm_permission_does_not_modify_env_or_leak_stderr(self):
        response = SimpleNamespace(
            returncode=254,
            stdout="",
            stderr="An error occurred (AccessDeniedException) fixture-secret",
        )
        with (
            patch.object(callback.subprocess, "run", return_value=response),
            self.assertRaisesRegex(RuntimeError, "AccessDeniedException") as failure,
        ):
            callback.configure(self.env_file, "/callback", "us-east-1")
        self.assertNotIn("fixture-secret", str(failure.exception))
        self.assertEqual(self.env_file.read_text(), self.original)

    def test_invalid_tokens_leave_env_unchanged(self):
        for value in (
            "",
            " \t ",
            "None",
            "null",
            None,
            42,
            "line\nbreak",
            "line\rbreak",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.configure(value)
            self.assertEqual(self.env_file.read_text(), self.original)

    def test_compose_preserves_special_characters(self):
        docker = shutil.which("docker")
        if (
            not docker
            or subprocess.run(
                [docker, "compose", "version"], capture_output=True, check=False
            ).returncode
        ):
            self.skipTest("Docker Compose CLI is unavailable")
        (self.directory / "compose.yaml").write_text(
            "services:\n  test:\n    image: alpine\n    environment:\n"
            "      TRAINING_CALLBACK_TOKEN: ${TRAINING_CALLBACK_TOKEN}\n"
        )
        environment = dict(os.environ)
        environment.pop("TRAINING_CALLBACK_TOKEN", None)
        for token in (
            "fixture-simple",
            "a$b'c\\d=.&|:value",
            "tail\\",
            'quote"${OTHER}',
        ):
            with self.subTest(token=token):
                self.configure(token)
                result = subprocess.run(
                    [
                        docker,
                        "compose",
                        "--project-directory",
                        str(self.directory),
                        "config",
                        "--environment",
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                actual = next(
                    line.partition("=")[2]
                    for line in result.stdout.splitlines()
                    if line.startswith("TRAINING_CALLBACK_TOKEN=")
                )
                self.assertEqual(actual, token)


if __name__ == "__main__":
    unittest.main()
