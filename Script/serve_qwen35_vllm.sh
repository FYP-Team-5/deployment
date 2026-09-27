#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements-vllm.txt}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
BOOTSTRAP_VENV="${BOOTSTRAP_VENV:-${SCRIPT_DIR}/.bootstrap}"
INSTALL_DEPS="${VLLM_INSTALL_DEPS:-1}"

# Both API model names share one base-model process. Selecting LORA_MODEL_NAME
# activates the adapter; selecting BASE_MODEL_NAME leaves it disabled.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-qwen35-9b-base}"
LORA_MODEL_NAME="${LORA_MODEL_NAME:-qwen35-9b-grading-qlora}"
LORA_PATH="${LORA_PATH:-SmuFypTeam5/GradingQlora}"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-16}"
QUANTIZATION="${VLLM_QUANTIZATION:-bitsandbytes}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VENV_DIR}/bin/vllm}"

usage() {
    cat <<'EOF'
Serve Qwen3.5-9B base and its grading LoRA through one vLLM server.

Usage:
  ./serve_qwen35_vllm.sh [extra vllm serve arguments]

API model names:
  qwen35-9b-base             Base model (adapter disabled)
  qwen35-9b-grading-qlora    Base model with grading adapter enabled

Important environment overrides:
  LORA_PATH                     Local adapter directory or Hugging Face repo ID
    VLLM_HOST                     Bind host (default: 0.0.0.0)
  VLLM_PORT                     Port (default: 8000)
    VLLM_MAX_MODEL_LEN            Context limit (default: 32768)
  VLLM_GPU_MEMORY_UTILIZATION   GPU fraction (default: 0.55)
  VLLM_MAX_NUM_SEQS             Concurrent sequences (default: 4)
  VLLM_QUANTIZATION             bitsandbytes or none (default: bitsandbytes)
  VLLM_ENFORCE_EAGER            1 saves CUDA-graph memory; 0 enables graphs
  VLLM_USE_FLASHINFER_SAMPLER   0 avoids nvcc JIT; 1 enables it (default: 0)
  VLLM_INSTALL_DEPS             Set to 0 to use an existing environment (default: 1)
  REQUIREMENTS_FILE             Pinned serving dependencies
  VENV_DIR                      Python 3.12 serving environment

Examples:
  ./serve_qwen35_vllm.sh
  LORA_PATH=SmuFypTeam5/GradingQlora ./serve_qwen35_vllm.sh
  VLLM_QUANTIZATION=none ./serve_qwen35_vllm.sh  # Opt out of 4-bit loading

After startup, list both models with:
  curl http://127.0.0.1:8000/v1/models

Use the model field in an OpenAI-compatible request to select either variant.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

install_dependencies() {
    [[ "$INSTALL_DEPS" == "1" ]] || {
        [[ "$INSTALL_DEPS" == "0" ]] || { echo "ERROR: VLLM_INSTALL_DEPS must be 0 or 1." >&2; exit 2; }
        return
    }

    [[ -f "$REQUIREMENTS_FILE" ]] || { echo "ERROR: missing $REQUIREMENTS_FILE" >&2; exit 1; }
    command -v python3 >/dev/null || { echo "ERROR: python3 is required for bootstrap." >&2; exit 1; }

    local requirements_hash stamp_file
    requirements_hash="$(sha256sum "$REQUIREMENTS_FILE" | cut -d' ' -f1)"
    stamp_file="${VENV_DIR}/.requirements-sha256"
    if [[ -x "$VLLM_BIN" && -f "$stamp_file" && "$(cat "$stamp_file")" == "$requirements_hash" ]]; then
        return
    fi

    if [[ ! -x "${BOOTSTRAP_VENV}/bin/uv" ]]; then
        python3 -m venv "$BOOTSTRAP_VENV"
        "${BOOTSTRAP_VENV}/bin/pip" install 'uv==0.9.16'
    fi

    local uv_bin="${BOOTSTRAP_VENV}/bin/uv"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        "$uv_bin" venv --python 3.12 --python-preference managed "$VENV_DIR"
    fi
    "$uv_bin" pip install --python "${VENV_DIR}/bin/python" -r "$REQUIREMENTS_FILE"
    printf '%s\n' "$requirements_hash" > "$stamp_file"
}

install_dependencies

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
    echo "ERROR: vLLM executable not found: $VLLM_BIN" >&2
    echo "Activate the CUDA-enabled environment that contains vLLM." >&2
    exit 1
fi

vllm_version="$($PYTHON_BIN -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))' 2>/dev/null || true)"
if [[ "$vllm_version" == *"+cpu"* ]]; then
    echo "ERROR: installed vLLM build is CPU-only ($vllm_version)." >&2
    echo "Install a CUDA-enabled vLLM build before serving Qwen3.5-9B." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "ERROR: the active Python environment does not have a working CUDA PyTorch runtime." >&2
    exit 1
fi

if [[ -d "$LORA_PATH" ]]; then
    if [[ ! -f "$LORA_PATH/adapter_config.json" || ! -f "$LORA_PATH/adapter_model.safetensors" ]]; then
        echo "ERROR: local adapter directory is incomplete: $LORA_PATH" >&2
        exit 1
    fi
elif [[ "$LORA_PATH" == /* || "$LORA_PATH" == ./* || "$LORA_PATH" == ../* ]]; then
    echo "ERROR: local adapter path does not exist: $LORA_PATH" >&2
    exit 1
fi

if [[ "$QUANTIZATION" != "none" && "$QUANTIZATION" != "bitsandbytes" ]]; then
    echo "ERROR: VLLM_QUANTIZATION must be 'none' or 'bitsandbytes'." >&2
    exit 2
fi

if [[ "$QUANTIZATION" == "bitsandbytes" ]]; then
    if ! "$PYTHON_BIN" -c 'import importlib.metadata; importlib.metadata.version("vllm-bnb-plugin")' >/dev/null 2>&1; then
        echo "ERROR: bitsandbytes mode requires the vllm-bnb-plugin package." >&2
        echo "Install it with: uv pip install vllm-bnb-plugin" >&2
        echo "To opt out of 4-bit loading, use VLLM_QUANTIZATION=none." >&2
        exit 1
    fi
fi

if [[ "$USE_FLASHINFER_SAMPLER" != "0" && "$USE_FLASHINFER_SAMPLER" != "1" ]]; then
    echo "ERROR: VLLM_USE_FLASHINFER_SAMPLER must be 0 or 1." >&2
    exit 2
fi

# FlashInfer sampling JIT-compiles CUDA code when its prebuilt kernel is not
# available. The PyTorch-native sampler works without a local nvcc toolkit.
export VLLM_USE_FLASHINFER_SAMPLER="$USE_FLASHINFER_SAMPLER"

vllm_args=(
    serve "$BASE_MODEL"
    --served-model-name "$BASE_MODEL_NAME"
    --enable-lora
    --lora-modules "${LORA_MODEL_NAME}=${LORA_PATH}"
    --max-lora-rank "$MAX_LORA_RANK"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --host "$HOST"
    --port "$PORT"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
    vllm_args+=(--enforce-eager)
elif [[ "$ENFORCE_EAGER" != "0" ]]; then
    echo "ERROR: VLLM_ENFORCE_EAGER must be 0 or 1." >&2
    exit 2
fi

if [[ "$QUANTIZATION" == "bitsandbytes" ]]; then
    vllm_args+=(--quantization bitsandbytes)
fi

echo "Starting one vLLM server with two selectable models:"
echo "  Base:    $BASE_MODEL_NAME ($BASE_MODEL)"
echo "  Adapter: $LORA_MODEL_NAME ($LORA_PATH)"
echo "  Quantization: $QUANTIZATION"
echo "  FlashInfer sampler: $USE_FLASHINFER_SAMPLER"
echo "  API:     http://${HOST}:${PORT}/v1"

exec "$VLLM_BIN" "${vllm_args[@]}" "$@"
