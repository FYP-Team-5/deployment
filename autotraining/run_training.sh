#!/usr/bin/env bash
# SSM executes this on the isolated GPU EC2 instance.
set -Eeuo pipefail
umask 077
BUCKET="$1"
REGION="$2"
CHECKSUM="$3"
[[ "$CHECKSUM" =~ ^[a-f0-9]{64}$ ]] || exit 2
WORK="/opt/fyp-autotraining"
MODEL_COMMIT="8a2b94ec95754d875dc546461ef15dc2a279ce5a"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$WORK/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"
cd "$WORK"
command -v nvidia-smi >/dev/null
nvidia-smi
if ! command -v aws >/dev/null || ! command -v gcc >/dev/null || ! command -v git >/dev/null; then
    apt-get update
    apt-get install -y awscli build-essential git python3-venv
fi
command -v python3 >/dev/null

aws s3 cp "s3://$BUCKET/scripts/train.py" train.py --region "$REGION" >/dev/null
aws s3 cp "s3://$BUCKET/scripts/requirements.txt" requirements.txt --region "$REGION" >/dev/null
aws s3 cp "s3://$BUCKET/datasets/$CHECKSUM/dataset.csv" "dataset.csv" --region "$REGION" >/dev/null

if [[ ! -d model/.git ]]; then
    git clone --filter=blob:none "https://github.com/FYP-Team-5/model.git" model
fi
git -C model fetch --depth 1 origin "$MODEL_COMMIT"
git -C model checkout --force FETCH_HEAD

if [[ ! -x .venv/bin/python ]]; then
    if ! python3 -m venv .venv; then
        apt-get update
        apt-get install -y python3-venv
        python3 -m venv .venv
    fi
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py \
    --bucket "$BUCKET" \
    --checksum "$CHECKSUM" \
    --dataset "$WORK/dataset.csv" \
    --notebook "$WORK/model/Rubrics/Qlora Comparison/qlora (48r,96a).ipynb" \
    --output "$WORK/output-$CHECKSUM"
