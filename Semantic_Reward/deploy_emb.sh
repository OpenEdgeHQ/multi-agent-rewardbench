#!/usr/bin/env bash
set -euo pipefail

EMBEDDING_MODEL_PATH="mnt/qwen3-emb-0.6b"
SERVICE_PORT=7000

export CUDA_VISIBLE_DEVICES=0,1

CMD=(vllm serve "$EMBEDDING_MODEL_PATH"
  --task embed
  --host 0.0.0.0
  --port "$SERVICE_PORT"
  --max-model-len 32768            
  --served-model-name Qwen3-Embedding-0.6B
  --trust-remote-code
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.9
)

echo "启动命令:"
printf ' %q' "${CMD[@]}"; echo

# 后台启动并写日志
nohup "${CMD[@]}" > embedding.log 2>&1 &

echo "启动 Qwen3 Embedding 服务... (日志: embedding.log)"
sleep 2
