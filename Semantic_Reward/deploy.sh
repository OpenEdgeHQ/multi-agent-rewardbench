#!/usr/bin/env bash
set -euo pipefail

EMBEDDING_MODEL_PATH="/mnt/Qwen3-Reranker-0.6B"
SERVICE_PORT=7000

export CUDA_VISIBLE_DEVICES=0,1


CMD=(vllm serve "$EMBEDDING_MODEL_PATH"
  --host 0.0.0.0
  --port "$SERVICE_PORT"
  --max-model-len 16384
  --served-model-name Qwen3-Reranker-0.6B
  --trust-remote-code
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.9
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
)

echo "启动命令:"
printf ' %q' "${CMD[@]}"; echo

# 后台启动并写日志
nohup "${CMD[@]}" > reranker.log 2>&1 &

echo "启动Reranker服务... (日志: reranker.log)"
sleep 2
