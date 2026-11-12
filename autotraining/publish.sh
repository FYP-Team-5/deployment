#!/usr/bin/env bash
# Runs after GPU stop on the existing backend host; retries never repeat training.
set -Eeuo pipefail
umask 077
bucket="$1"
export AWS_DEFAULT_REGION="$2"
work=/opt/fyp-training-publish
mkdir -p "$work"
if [[ ! -x "$work/.venv/bin/python" ]]; then
    python3 -m venv "$work/.venv"
fi
"$work/.venv/bin/python" -m pip install --quiet 'boto3>=1.40,<2' 'huggingface_hub>=0.35,<2'
aws s3 cp "s3://$bucket/scripts/publish.py" "$work/publish.py" >/dev/null
"$work/.venv/bin/python" "$work/publish.py" --bucket "$bucket" --checksum "$3" --run-id "$4" \
    --repo "$5" --hf-token-parameter "$6" --callback-token-parameter "$7" --callback-url "$8" \
    --feedback-repo "${9:-SmuFypTeam5/feedback}"
