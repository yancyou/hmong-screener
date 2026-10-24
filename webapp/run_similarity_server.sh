#!/usr/bin/env bash
# Start the Hmong audio similarity API server (no xlsr_300m.pt needed —
# the model architecture is reconstructed from the checkpoint itself).
#
# Usage:
#   ./run_similarity_server.sh                          # default ckpt, layer 20
#   ./run_similarity_server.sh <checkpoint.pt>          # switch checkpoint
#   ./run_similarity_server.sh <checkpoint.pt> <layer>  # layer spec: 20 or 19,20,21
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# Uses whatever python is active (activate your conda env first).
# Override with:  PYTHON=/path/to/python ./run_similarity_server.sh ...
PYTHON="${PYTHON:-python}"

CKPT="${1:-$HERE/checkpoint_slim.pt}"
# Default: final encoder output. Pass a spec like 20 or 19,20,21 to override.
LAYER="${2:--1}"

# Allow passing a bare filename relative to this repo (absolute paths as-is)
[[ "$CKPT" = /* ]] || CKPT="$HERE/$CKPT"
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "Checkpoint : $CKPT"
echo "Layer      : $([ "$LAYER" = "-1" ] && echo final || echo "$LAYER")"
echo "Server     : http://127.0.0.1:${PORT:-8000}  (healthz: /healthz)"

cd "$HERE"
CHECKPOINT_PATH="$CKPT" \
EMBEDDING_LAYER="$LAYER" \
HOST="${HOST:-0.0.0.0}" \
PORT="${PORT:-8000}" \
exec "$PYTHON" api_similarity_server.py
