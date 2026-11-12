"""Wait for vLLM startup before systemd reports the serving service ready."""

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def probe(base_url, model_names, runtime_enabled, request_timeout=5):
    headers = {}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

    def get(path, *, as_json=True):
        request = Request(base_url.rstrip("/") + path, headers=headers)
        try:
            response = urlopen(request, timeout=request_timeout)
        except HTTPError as error:
            error.close()
            raise
        with response:
            return json.load(response) if as_json else None

    get("/health", as_json=False)
    available = {item["id"] for item in get("/v1/models")["data"]}
    missing = set(model_names) - available
    if missing:
        raise ValueError("Missing model aliases: " + ", ".join(sorted(missing)))
    if runtime_enabled:
        paths = get("/openapi.json")["paths"]
        if "post" not in paths.get("/v1/load_lora_adapter", {}):
            raise ValueError("Runtime adapter loading API is unavailable")


def wait_until_ready(
    base_url, model_names, runtime_enabled, *, timeout=2400, interval=5
):
    deadline = time.monotonic() + timeout
    last_error = "No readiness response"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"vLLM startup timed out: {last_error}")
        try:
            probe(base_url, model_names, runtime_enabled, min(5, remaining))
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = str(error)
            if reason != last_error:
                print(f"Waiting for vLLM: {reason}", flush=True)
            last_error = reason
            time.sleep(max(0, min(interval, deadline - time.monotonic())))
        else:
            print(
                "vLLM ready: configured models and required APIs are available.",
                flush=True,
            )
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    host = os.environ.get("FYP_VLLM_HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1" if host == "0.0.0.0" else "[::1]"
    elif ":" in host:
        host = f"[{host}]"
    parser.add_argument(
        "--url", default=f"http://{host}:" + os.environ.get("FYP_VLLM_PORT", "8000")
    )
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    model_names = [
        os.environ.get("BASE_MODEL_NAME", "qwen35-9b-base"),
        os.environ.get("LORA_MODEL_NAME", "qwen35-9b-grading-qlora"),
        os.environ.get("FEEDBACK_LORA_MODEL_NAME", "qwen35-9b-feedback-qlora"),
    ]
    try:
        wait_until_ready(
            args.url,
            model_names,
            os.environ.get("FYP_LORA_RUNTIME_ENABLED", "false") == "true",
            timeout=args.timeout,
        )
    except TimeoutError as error:
        print(
            f"ERROR: {error}. Check journalctl -u qwen-vllm.service.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
