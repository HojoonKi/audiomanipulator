#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="audiomanipulator:latest"
CONTAINER_NAME="audiomanipulator"

# 빌드
docker build -t "${IMAGE_NAME}" .

# 기존 컨테이너 정리
if docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# 런옵션 구성 (compose 대체)
docker run \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  -it \
  --env PYTHONPATH=/app \
  --env HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-} \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --env TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0} \
  --env HF_LOCAL_ONLY=${HF_LOCAL_ONLY:-0} \
  --env SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers \
  --env CLAP_SKIP_DOWNLOAD=${CLAP_SKIP_DOWNLOAD:-0} \
  --env CLAP_CKPT_PATH=${CLAP_CKPT_PATH:-} \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)":/app \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/sentence-transformers":/root/.cache/sentence-transformers \
  -v "$(pwd)/checkpoints":/app/checkpoints \
  -v "$(pwd)/output":/app/output \
  -v "$(pwd)/audio_dataset":/app/audio_dataset \
  -w /app \
  "${IMAGE_NAME}" bash


