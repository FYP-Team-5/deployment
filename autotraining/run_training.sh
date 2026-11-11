#!/usr/bin/env bash
# SSM executes this on the isolated GPU EC2 instance.
set -Eeuo pipefail
umask 077
BUCKET="$1"
REGION="$2"
CHECKSUM="$3"
VERSION_ID="$4"
RUN_ID="$5"
[[ "$RUN_ID" =~ ^[a-f0-9]{32}$ ]] || exit 2
[[ "$VERSION_ID" =~ ^[a-f0-9-]{36}$ ]] || exit 2
[[ "$CHECKSUM" =~ ^[a-f0-9]{64}$ ]] || exit 2
# Keep training state on the provisioned 200 GiB EBS root volume. The instance
# store and its DLAMI mount may change or disappear when EC2 is stopped/resized.
WORK="${FYP_TRAINING_WORK_DIRECTORY:-/opt/fyp-autotraining}"
[[ "$WORK" == /* && "$WORK" != / ]] || { echo 'ERROR: Training workspace must be an absolute non-root path.' >&2; exit 2; }
MODEL_COMMIT="8a2b94ec95754d875dc546461ef15dc2a279ce5a"
export AWS_DEFAULT_REGION="$REGION"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"
# Defend against duplicate SSM delivery or a redriven Launch state.
exec 9>"$WORK/training.lock"
flock -n 9 || { echo 'A training process already owns this GPU' >&2; exit 3; }
[[ ! -f "$WORK/completed-$RUN_ID" ]] || exit 0

# Budget for the base model, Python/CUDA packages, caches and checkpoints before
# downloading anything large. The check uses actual free space, not disk names
# or the instance type's advertised NVMe capacity.
min_free_gib="${FYP_TRAINING_MIN_FREE_GIB:-60}"
[[ "$min_free_gib" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: FYP_TRAINING_MIN_FREE_GIB must be a positive integer.' >&2; exit 2; }
available_kib="$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || { echo 'ERROR: Cannot determine free workspace storage.' >&2; exit 1; }
if (( available_kib < min_free_gib * 1024 * 1024 )); then
    echo "ERROR: Training workspace $WORK needs at least $min_free_gib GiB free; available: $available_kib KiB. Check the EBS volume/filesystem or clean old caches." >&2
    exit 1
fi

export HF_HOME="$WORK/hf-cache"
export PIP_CACHE_DIR="$WORK/pip-cache"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export XDG_CACHE_HOME="$WORK/cache"
export TMPDIR="$WORK/tmp-$RUN_ID"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
trap 'rm -rf -- "$WORK/output-$RUN_ID" "$WORK/feedback-output-$RUN_ID" "$TMPDIR"' EXIT
echo "Training workspace (persistent EBS): $WORK"
df -h "$WORK"
# Emergency cost guard if a Step Functions execution is force-terminated.
# EC2 is configured to stop rather than terminate on guest shutdown.
if command -v systemd-run >/dev/null; then
    systemd-run --unit "fyp-trainer-watchdog-$RUN_ID" \
        --on-active=15h /sbin/shutdown -h now || true
fi
cd "$WORK"
command -v nvidia-smi >/dev/null
nvidia-smi
if ! command -v aws >/dev/null || ! command -v gcc >/dev/null || ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    apt-get update
    apt-get install -y awscli build-essential git curl ca-certificates
fi
command -v python3 >/dev/null

aws s3 cp "s3://$BUCKET/scripts/train.py" train.py --region "$REGION" >/dev/null
aws s3 cp "s3://$BUCKET/scripts/train_feedback.py" train_feedback.py --region "$REGION" >/dev/null
aws s3 cp "s3://$BUCKET/scripts/requirements.txt" requirements.txt --region "$REGION" >/dev/null
aws s3 cp "s3://$BUCKET/datasets/$CHECKSUM/dataset.csv" "dataset.csv" --region "$REGION" >/dev/null

if [[ ! -d model/.git ]]; then
    git clone --filter=blob:none "https://github.com/FYP-Team-5/model.git" model
fi
git -C model fetch --depth 1 origin "$MODEL_COMMIT"
git -C model checkout --force FETCH_HEAD

# Ubuntu 22.04 defaults to Python 3.10; Transformers 5 requires >=3.11.
# Use a dedicated Python 3.12 environment without altering the serving host.
if [[ ! -x "$WORK/bin/uv" ]]; then
    curl -LsSf --retry 3 https://astral.sh/uv/0.9.10/install.sh -o "$WORK/install-uv.sh"
    UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh "$WORK/install-uv.sh"
fi
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'; then
    "$WORK/bin/uv" venv --clear --seed --python 3.12 .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python train.py \
    --grading-only \
    --bucket "$BUCKET" \
    --checksum "$CHECKSUM" \
    --run-id "$RUN_ID" \
    --dataset "$WORK/dataset.csv" \
    --version-id "$VERSION_ID" \
    --notebook "$WORK/model/Rubrics/Qlora Comparison/qlora (48r,96a).ipynb" \
    --output "$WORK/output-$RUN_ID"

# A separate process releases the grading model, optimizer and CUDA allocations.
.venv/bin/python train_feedback.py \
    --bucket "$BUCKET" \
    --checksum "$CHECKSUM" \
    --run-id "$RUN_ID" \
    --dataset "$WORK/dataset.csv" \
    --version-id "$VERSION_ID" \
    --output "$WORK/feedback-output-$RUN_ID"

# Candidate is durable in S3; redriving this same run does not train twice.
touch "$WORK/completed-$RUN_ID"
