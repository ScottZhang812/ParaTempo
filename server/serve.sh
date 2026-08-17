#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
CUDA_DEVICE="${CUDA_DEVICE:-GPU_ID_XXX}"
PORT="${PORT:-PORT_XXX}"
API_KEY="${API_KEY:-API_KEY_XXX}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-vllm_logs}"

if [ "${CUDA_DEVICE}" = "GPU_ID_XXX" ]; then
  echo "Set CUDA_DEVICE to the local GPU id before running."
  exit 2
fi
if [ "${PORT}" = "PORT_XXX" ]; then
  echo "Set PORT to the local vLLM port before running."
  exit 2
fi
if [ "${API_KEY}" = "API_KEY_XXX" ]; then
  echo "Set API_KEY to the local vLLM API key before running."
  exit 2
fi
case "${MODEL}" in
  "Qwen/Qwen3.5-35B-A3B"|"openai/gpt-oss-20b")
    ;;
  *)
    echo "Unsupported MODEL. Use Qwen/Qwen3.5-35B-A3B or openai/gpt-oss-20b."
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"

TEMPLATE_ARG=()
if [[ "${MODEL}" == *"Qwen"* ]] || [[ "${MODEL}" == *"qwen"* ]]; then
  TEMPLATE_PATH="$(dirname "$0")/template/qwen3.jinja"
  if [ -f "${TEMPLATE_PATH}" ]; then
    TEMPLATE_ARG=(--chat-template "${TEMPLATE_PATH}")
  fi
fi

if command -v lsof >/dev/null 2>&1 && lsof -ti tcp:"${PORT}" >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Choose another PORT_XXX value."
  exit 2
fi

LOG_FILE="${LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).out"
echo "Starting vLLM model=${MODEL} cuda_device=${CUDA_DEVICE} port=${PORT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" nohup "${VLLM_BIN}" serve "${MODEL}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu_memory_utilization "${GPU_MEM}" \
  --api-key "${API_KEY}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  "${TEMPLATE_ARG[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo "PID: $! | Log: ${LOG_FILE}"
