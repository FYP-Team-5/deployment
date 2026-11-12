"""Exercise the serving wrapper without downloading models or requiring a GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "serve_qwen35_vllm.sh"


class ServingWrapperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        python_stub = self.directory / "python-stub"
        python_stub.write_text("#!/bin/bash\nexit 0\n")
        python_stub.chmod(0o755)
        vllm_stub = self.directory / "vllm-stub"
        vllm_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "expected = os.environ.get('TEST_EXPECT_HF_TOKEN')\n"
            "if expected and os.environ.get('HF_TOKEN') != expected:\n"
            "    sys.exit('Expected HF token was not passed to vLLM')\n"
            "if os.environ.get('TEST_EXPECT_RUNTIME') == '1':\n"
            "    assert os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] == 'True'\n"
            "    plugins = os.environ['VLLM_PLUGINS'].split(',')\n"
            "    assert 'bitsandbytes' in plugins, 'Quantization plugin was excluded'\n"
            "    assert 'lora_filesystem_resolver' in plugins\n"
            "    assert os.path.isdir(os.environ['VLLM_LORA_RESOLVER_CACHE_DIR'])\n"
            "print(json.dumps(sys.argv[1:]))\n"
        )
        vllm_stub.chmod(0o755)
        aws_stub = self.directory / "aws"
        aws_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_SSM_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
            "if os.environ.get('TEST_SSM_ERROR') == '1':\n"
            "    print('Simulated AccessDenied', file=sys.stderr)\n"
            "    sys.exit(42)\n"
            "print(os.environ.get('TEST_SSM_TOKEN', 'fixture-token-from-ssm'))\n"
        )
        aws_stub.chmod(0o755)
        for name in ("grading", "feedback"):
            adapter = self.directory / name
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").touch()
        self.environment = {
            **os.environ,
            "PATH": str(self.directory)
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "HF_TOKEN": "",
            "HF_TOKEN_SSM_PARAMETER": "",
            "HF_TOKEN_SSM_REGION": "us-east-1",
            "TEST_SSM_LOG": str(self.directory / "ssm-request.json"),
            "TEST_SSM_ERROR": "0",
            "TEST_EXPECT_HF_TOKEN": "",
            "FYP_VLLM_INSTALL_DEPS": "0",
            "FYP_VLLM_QUANTIZATION": "none",
            "FYP_LORA_RUNTIME_ENABLED": "false",
            "FYP_VLLM_MAX_LORAS": "1",
            "FYP_VLLM_MAX_CPU_LORAS": "4",
            "FYP_VLLM_MAX_LORA_RANK": "64",
            "BASE_MODEL": "Qwen/Qwen3.5-9B",
            "BASE_MODEL_NAME": "qwen35-9b-base",
            "LORA_MODEL_NAME": "qwen35-9b-grading-qlora",
            "FEEDBACK_LORA_MODEL_NAME": "qwen35-9b-feedback-qlora",
            "LORA_PATH": str(self.directory / "grading"),
            "FEEDBACK_LORA_PATH": str(self.directory / "feedback"),
            "PYTHON_BIN": str(python_stub),
            "VLLM_BIN": str(vllm_stub),
        }

    def run_wrapper(self, *, trace=False, **overrides):
        return subprocess.run(
            ["bash", *(["-x"] if trace else []), str(SCRIPT)],
            env={**self.environment, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_one_base_process_serves_both_adapter_aliases(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[:2], ["serve", "Qwen/Qwen3.5-9B"])
        index = args.index("--lora-modules")
        self.assertEqual(
            args[index + 1 : index + 3],
            [
                f"qwen35-9b-grading-qlora={self.directory}/grading",
                f"qwen35-9b-feedback-qlora={self.directory}/feedback",
            ],
        )
        self.assertEqual(args[args.index("--max-lora-rank") + 1], "64")
        self.assertEqual(args[args.index("--max-loras") + 1], "1")
        self.assertEqual(args[args.index("--max-cpu-loras") + 1], "4")

    def test_multiple_adapter_batches_can_be_enabled_explicitly(self):
        result = self.run_wrapper(FYP_VLLM_MAX_LORAS="2", FYP_VLLM_MAX_NUM_SEQS="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[args.index("--max-loras") + 1], "2")

    def test_runtime_loading_keeps_bitsandbytes_quantization_available(self):
        result = self.run_wrapper(
            FYP_LORA_RUNTIME_ENABLED="true",
            FYP_LORA_ADAPTER_DIRECTORY=str(self.directory / "candidates"),
            FYP_VLLM_QUANTIZATION="bitsandbytes",
            TEST_EXPECT_RUNTIME="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(args[args.index("--quantization") + 1], "bitsandbytes")

    def test_invalid_adapter_capacity_is_rejected(self):
        for overrides in (
            {"FYP_VLLM_MAX_CPU_LORAS": "1"},
            {"FYP_VLLM_MAX_LORAS": "5"},
            {"FYP_VLLM_MAX_LORAS": "zero"},
            {"FYP_VLLM_MAX_CPU_LORAS": "0"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_wrapper(**overrides)
                self.assertEqual(result.returncode, 2, result.stderr)

    def test_aliases_must_be_distinct(self):
        result = self.run_wrapper(FEEDBACK_LORA_MODEL_NAME="qwen35-9b-grading-qlora")
        self.assertEqual(result.returncode, 2)
        self.assertIn("aliases must be distinct", result.stderr)

    def test_missing_feedback_adapter_directory_is_rejected(self):
        result = self.run_wrapper(FEEDBACK_LORA_PATH=str(self.directory / "missing"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("local adapter path does not exist", result.stderr)

    def test_ssm_token_is_exported_to_vllm_without_logging_its_value(self):
        token = "fixture-secret-from-ssm"
        result = self.run_wrapper(
            HF_TOKEN_SSM_PARAMETER="/fyp/serving/hf-token",
            TEST_SSM_TOKEN=token,
            TEST_EXPECT_HF_TOKEN=token,
            HF_TOKEN="stale-direct-token",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        request = json.loads((self.directory / "ssm-request.json").read_text())
        self.assertEqual(request[:2], ["ssm", "get-parameter"])
        self.assertEqual(request[request.index("--name") + 1], "/fyp/serving/hf-token")
        self.assertIn("--with-decryption", request)
        self.assertEqual(request[request.index("--region") + 1], "us-east-1")
        self.assertNotIn(token, result.stdout + result.stderr)
        self.assertIn("Serving HF token retrieved from SSM parameter", result.stdout)

    def test_ssm_token_is_not_leaked_under_shell_tracing(self):
        token = "fixture-secret-hidden-from-trace"
        result = self.run_wrapper(
            trace=True,
            HF_TOKEN_SSM_PARAMETER="/fyp/serving/hf-token",
            TEST_SSM_TOKEN=token,
            TEST_EXPECT_HF_TOKEN=token,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(token, result.stdout + result.stderr)

    def test_ssm_failure_does_not_fall_back_to_a_stale_direct_token(self):
        result = self.run_wrapper(
            HF_TOKEN_SSM_PARAMETER="/fyp/serving/hf-token",
            TEST_SSM_ERROR="1",
            HF_TOKEN="stale-direct-token",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot retrieve the serving HF token", result.stderr)
        self.assertIn("parameter '/fyp/serving/hf-token'", result.stderr)
        self.assertIn("region: us-east-1", result.stderr)
        self.assertIn("AWS CLI exit: 42", result.stderr)
        self.assertNotIn("Starting one vLLM", result.stdout)

    def test_empty_ssm_tokens_are_rejected(self):
        for token in ("", " \n ", "None", "null"):
            with self.subTest(token=token):
                result = self.run_wrapper(
                    HF_TOKEN_SSM_PARAMETER="/fyp/serving/hf-token",
                    TEST_SSM_TOKEN=token,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("SSM serving HF token is empty", result.stderr)

    def test_direct_token_still_works_without_ssm_or_an_env_file(self):
        result = self.run_wrapper(
            HF_TOKEN="fixture-direct-token", TEST_EXPECT_HF_TOKEN="fixture-direct-token"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.directory / "ssm-request.json").exists())


if __name__ == "__main__":
    unittest.main()
