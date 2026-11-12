"""Exercise bootstrap recovery without systemd or an AWS instance."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "Script/systemctl_retry.sh"


class BootstrapRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.log = self.directory / "calls"
        self.environment = {
            **os.environ,
            "PATH": str(self.directory) + os.pathsep + os.environ["PATH"],
            "TEST_CALLS": str(self.log),
            "TEST_STARTED": str(self.directory / "started"),
            "TEST_FAILURE": "",
        }
        self.stub(
            "systemctl",
            """printf '%s\\n' "systemctl $*" >> "$TEST_CALLS"
if [[ "$*" == 'enable --now qwen-vllm.service' ]]; then
    if [[ "$TEST_FAILURE" == 'service' ]]; then
        echo 'Job for qwen-vllm.service failed: control process exited with error code.' >&2
        exit 1
    fi
    if [[ "$TEST_FAILURE" == 'bus' ]] || [[ ! -e "$TEST_STARTED" ]]; then
        touch "$TEST_STARTED"
        echo 'Warning! D-Bus connection terminated.' >&2
        echo 'Failed to wait for response: Connection reset by peer' >&2
        exit 1
    fi
fi
if [[ "$*" == 'is-active --quiet qwen-vllm.service' && "$TEST_FAILURE" == 'inactive' ]]; then
    exit 3
fi
if [[ "$*" == 'is-active --quiet amazon-cloudwatch-agent.service' && "$TEST_FAILURE" == 'logging' ]]; then
    exit 3
fi
""",
        )
        self.stub("sleep", 'echo "sleep $*" >> "$TEST_CALLS"\n')
        self.stub("bootstrap-fyp-apps", 'echo apps >> "$TEST_CALLS"\n')

    def stub(self, name, body):
        path = self.directory / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_bootstrap(self, failure=""):
        # Run the actual final bootstrap sequence with only external commands stubbed.
        tail = (
            (ROOT / "user-data.sh.tftpl")
            .read_text()
            .split("# Start host services.", 1)[1]
            .split("\n", 1)[1]
        )
        tail = tail.replace("/usr/local/sbin/bootstrap-fyp-apps", "bootstrap-fyp-apps")
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -Eeuo pipefail\nsource "$1"\n' + tail,
                "bash",
                str(HELPER),
            ],
            env={**self.environment, "TEST_FAILURE": failure},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result, self.log.read_text().splitlines()

    def test_disconnect_rejoins_start_and_bootstraps_after_readiness_and_logging(self):
        result, calls = self.run_bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("systemctl enable --now qwen-vllm.service"), 2)
        self.assertEqual(calls.count("sleep 5"), 1)
        self.assertEqual(
            calls[-4:],
            [
                "systemctl is-active --quiet qwen-vllm.service",
                "systemctl restart amazon-cloudwatch-agent.service",
                "systemctl is-active --quiet amazon-cloudwatch-agent.service",
                "apps",
            ],
        )

    def test_persistent_disconnect_stops_after_six_attempts(self):
        result, calls = self.run_bootstrap("bus")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(calls.count("systemctl enable --now qwen-vllm.service"), 6)
        self.assertEqual(calls.count("sleep 5"), 5)
        self.assertNotIn("apps", calls)

    def test_real_service_failure_is_not_retried(self):
        result, calls = self.run_bootstrap("service")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(calls.count("systemctl enable --now qwen-vllm.service"), 1)
        self.assertNotIn("sleep 5", calls)
        self.assertNotIn("apps", calls)

    def test_inactive_service_does_not_bootstrap_apps(self):
        result, calls = self.run_bootstrap("inactive")
        self.assertEqual(result.returncode, 3)
        self.assertNotIn("apps", calls)
        self.assertNotIn("systemctl restart amazon-cloudwatch-agent.service", calls)

    def test_failed_logging_check_does_not_bootstrap_apps(self):
        result, calls = self.run_bootstrap("logging")
        self.assertEqual(result.returncode, 3)
        self.assertNotIn("apps", calls)


if __name__ == "__main__":
    unittest.main()
