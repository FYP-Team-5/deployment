"""Configure the Compose backend callback credential from an existing SSM secret."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def configure(env_file, parameter, region):
    result = subprocess.run(
        [
            "aws",
            "ssm",
            "get-parameter",
            "--name",
            parameter,
            "--with-decryption",
            "--region",
            region,
            "--output",
            "json",
            "--cli-connect-timeout",
            "5",
            "--cli-read-timeout",
            "10",
        ],
        env={**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if result.returncode:
        match = re.search(r"An error occurred \(([A-Za-z0-9]+)\)", result.stderr)
        reason = match[1] if match else f"AWS CLI exit {result.returncode}"
        raise RuntimeError(
            f"Cannot read callback token from {parameter} in {region}: {reason}"
        )
    try:
        value = json.loads(result.stdout)["Parameter"]["Value"]
    except (ValueError, KeyError, TypeError):
        raise ValueError(
            "SSM returned an invalid callback parameter response"
        ) from None
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in ("None", "null")
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("SSM callback token must be a nonempty single-line credential")
    # Compose double-quoted values support JSON escapes; $$ prevents interpolation.
    entry = "TRAINING_CALLBACK_TOKEN=" + json.dumps(value, ensure_ascii=False).replace(
        "$", "$$"
    )
    path = Path(env_file)
    lines = path.read_text().splitlines()
    lines = [
        line
        for line in lines
        if not re.match(r"\s*(?:export\s+)?TRAINING_CALLBACK_TOKEN\s*[=:]", line)
    ]
    content = "\n".join([*lines, entry]) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as target:
            temporary = Path(target.name)
            target.write(content)
        # NamedTemporaryFile creates a private 0600 file; replace atomically.
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    try:
        configure(args.env_file, args.parameter, args.region)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: Callback configuration failed: {error}", file=sys.stderr)
        return 1
    print("Backend training callback credential configured from SSM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
