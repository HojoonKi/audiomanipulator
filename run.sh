#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="audiomanipulator:latest"
CONTAINER_NAME="audiomanipulator"

# 옵션: HF 토큰이 설정된 경우에만 전달
HF_TOKEN_ARG=""
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  HF_TOKEN_ARG="--env HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}"
fi

# 옵션: MASTER_PORT가 설정된 경우에만 전달
MASTER_PORT_ARG=""
if [ -n "${MASTER_PORT:-}" ]; then
  MASTER_PORT_ARG="--env MASTER_PORT=${MASTER_PORT}"
fi

# 빌드
docker build -t "${IMAGE_NAME}" .

# 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# 런옵션 구성 (compose 대체)
docker run \
  --gpus all \
  --ipc=host \
  --name "${CONTAINER_NAME}" \
  -it \
  --shm-size=64g \
  --ipc=host \
  --ulimit memlock=-1 \
  --env PYTHONPATH=/app \
  --env TORCH_HOME=/root/.cache/torch \
  --env HF_HOME=/root/.cache/huggingface \
  ${HF_TOKEN_ARG} \
  ${MASTER_PORT_ARG} \
  --env NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
  --env NCCL_TIMEOUT=${NCCL_TIMEOUT:-120} \
  --env CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1} \
  --env TORCH_USE_CUDA_DSA=1 \
  --env TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0} \
  --env HF_LOCAL_ONLY=${HF_LOCAL_ONLY:-0} \
  --env SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers \
  --env CLAP_CACHE_DIR=/root/.cache/laion_clap \
  --env CLAP_SKIP_DOWNLOAD=${CLAP_SKIP_DOWNLOAD:-0} \
  --env CLAP_CKPT_PATH=${CLAP_CKPT_PATH:-} \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)":/app \
  -v "$HOME/.cache/laion_clap":/root/.cache/laion_clap \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/sentence-transformers":/root/.cache/sentence-transformers \
  -v "$(pwd)/checkpoints":/app/checkpoints \
  -v "$(pwd)/output":/app/output \
  -v "$(pwd)/audio_dataset":/app/audio_dataset \
  -w /app \
  "${IMAGE_NAME}" bash -lc 'if [ -d "/app/dasp-pytorch" ]; then cd /app/dasp-pytorch && pip install -e . && cd /app; fi; exec bash'


