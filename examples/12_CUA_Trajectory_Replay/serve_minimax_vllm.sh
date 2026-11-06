#!/usr/bin/env bash
# Serve MiniMax M3.1 Flash with vLLM on one 4-GPU Blackwell node, using the
# official recipe's single-node command
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash, GB300, single_node_tp).
#
# Run inside the vLLM image (vllm/vllm-openai:minimax-m31-flash; vLLM >= 0.33).
# Extra arguments are appended to `vllm serve`, e.g. --max-model-len 262144.
#
#   MODEL_DIR=/path/to/MiniMax-M3.1-Flash ./serve_minimax_vllm.sh
#
#   TP=4            tensor-parallel size
#   SPEC_DECODE=1   built-in DSpark speculative decoding (recipe default); 0 disables it
#   PORT=8000
#
# Image input must stay enabled: do not use --language-model-only or the
# recipe's text_only feature, which the recorded CUA requests need.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the MiniMax-M3.1-Flash checkpoint directory}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
SPEC_DECODE="${SPEC_DECODE:-1}"

spec_args=()
if [[ "${SPEC_DECODE}" == 1 ]]; then
  spec_args=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_TS","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}')
fi

export VLLM_FLOAT32_MATMUL_PRECISION=high
exec vllm serve "${MODEL_DIR}" \
  --served-model-name MiniMaxAI/MiniMax-M3.1-Flash \
  --block-size 128 \
  --disable-hybrid-kv-cache-manager \
  --tensor-parallel-size "${TP}" \
  --kv-cache-dtype nvfp4 \
  --attention-config '{"indexer_kv_dtype": "nvfp4", "use_prefill_query_quantization": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  "${spec_args[@]}" \
  --enable-prompt-tokens-details \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "$@"
