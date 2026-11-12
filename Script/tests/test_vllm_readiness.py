"""Exercise startup readiness against a local vLLM-shaped HTTP server."""

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_vllm_ready.py"
SPEC = importlib.util.spec_from_file_location("check_vllm_ready", SCRIPT)
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.models = ["base", "grading", "feedback"]
        self.health_status = 200
        self.health_requests = 0
        self.paths = {"/v1/load_lora_adapter": {"post": {}}}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status = 200
                if self.path == "/health":
                    owner.health_requests += 1
                    status = owner.health_status
                    payload = {}
                elif self.path == "/v1/models":
                    payload = {"data": [{"id": name} for name in owner.models]}
                elif self.path == "/openapi.json":
                    payload = {"paths": owner.paths}
                else:
                    status, payload = 404, {}
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *_):
                pass

        # HTTPServer's reverse-DNS lookup is unnecessary for a loopback fixture.
        with patch("socket.getfqdn", return_value="localhost"):
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def test_ready_with_all_models_and_runtime_api(self):
        readiness.probe(self.url, self.models, True)

    def test_engine_health_failure_is_not_ready(self):
        self.health_status = 503
        with self.assertRaises(OSError):
            readiness.probe(self.url, self.models, True)

    def test_missing_feedback_model_is_not_ready(self):
        self.models.remove("feedback")
        with self.assertRaisesRegex(ValueError, "Missing model aliases: feedback"):
            readiness.probe(self.url, ["base", "grading", "feedback"], True)

    def test_missing_runtime_api_is_not_ready(self):
        self.paths = {}
        with self.assertRaisesRegex(ValueError, "Runtime adapter loading API"):
            readiness.probe(self.url, self.models, True)

    def test_runtime_api_is_optional_when_disabled(self):
        self.paths = {}
        readiness.probe(self.url, self.models, False)

    def test_retries_until_engine_becomes_healthy(self):
        self.health_status = 503

        def become_healthy(_):
            self.health_status = 200

        with patch.object(readiness.time, "sleep", side_effect=become_healthy):
            readiness.wait_until_ready(self.url, self.models, True, timeout=1)
        self.assertEqual(self.health_requests, 2)

    def test_unavailable_runtime_api_times_out_with_reason(self):
        self.paths = {}
        with self.assertRaisesRegex(TimeoutError, "Runtime adapter loading API"):
            readiness.wait_until_ready(
                self.url, self.models, True, timeout=0.05, interval=0.01
            )


if __name__ == "__main__":
    unittest.main()
