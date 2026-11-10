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
            f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
        )
        vllm_stub.chmod(0o755)
        for name in ("grading", "feedback"):
            adapter = self.directory / name
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").touch()
        self.environment = {
            **os.environ,
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

    def run_wrapper(self, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT)],
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


if __name__ == "__main__":
    unittest.main()
