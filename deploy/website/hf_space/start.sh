#!/bin/bash
# Download the private bundle (code + screening model) on first start, then run the server.
set -eo pipefail
APP="${APP_DIR:-/tmp/app}"
if [ ! -f "$APP/api_similarity_server.py" ]; then
  mkdir -p "$APP"
  if [ -n "${BUNDLE_FILE:-}" ]; then
    echo "[start] unpacking local bundle $BUNDLE_FILE"; tar -xf "$BUNDLE_FILE" -C "$APP"
  else
    : "${PRIVATE_REPO:?set the Space variable PRIVATE_REPO}"; : "${HF_TOKEN:?add the Space secret HF_TOKEN}"
    echo "[start] downloading private bundle from $PRIVATE_REPO"
    curl -fL --retry 5 -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/$PRIVATE_REPO/resolve/main/bundle.tar" | tar -x -C "$APP"
  fi
fi
# The token is only needed for the download; keep it out of the public-facing server's environment.
unset HF_TOKEN
export PYTHONPATH="$APP/fairseq:${PYTHONPATH:-}"
export CTC_CHECKPOINT_PATH="$APP/ctc_checkpoint.pt" CTC_VOCAB="$APP/vocab.csv"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$APP"
echo "[start] starting server"
exec python api_similarity_server.py
