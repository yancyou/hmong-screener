#!/bin/bash
# Build the website's private bundle (bundle.tar): web app (webapp/) + our patches + scorer files + model.
# The public Space downloads this file from the private repo at startup (see hf_space/start.sh).
#   bash deploy/website/build_bundle.sh [output.tar]
# Needs Node 20 / npm on PATH (on CHTC: export PATH=/home/xouyang7/hmong/_swtest/node/bin:$PATH) and Python 3.
# HMONG_MODEL: the deployed model, on staging or downloaded from the v1.0 release (README, "Model").
# HMONG_UPSTREAM: the web app, a folder (default webapp/) or the archived .tar.gz on staging.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGING="${HMONG_STAGING:-/staging/x/xouyang7}"
UPSTREAM="${HMONG_UPSTREAM:-$ROOT/webapp}"
MODEL="${HMONG_MODEL:-$STAGING/full32_ckpts/full32_foldfinal_infer.pt}"
OUT="${1:-$ROOT/build/hf_bundle.tar}"
command -v npm >/dev/null || { echo "npm not found: put Node 20 on PATH"; exit 1; }
for f in "$UPSTREAM" "$MODEL"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
W="$ROOT/build/webapp"; rm -rf "$W"; mkdir -p "$W" "$(dirname "$OUT")"; APP="$W/hmong_audio_software-main"
if [ -d "$UPSTREAM" ]; then cp -a "$UPSTREAM" "$APP"; else tar -xzf "$UPSTREAM" -C "$W"; fi
cp "$ROOT"/deploy/scorer/{ctc_scorer.py,f0_infer.py} "$ROOT"/deploy/patches/{ctc_endpoint_patch.py,gui_ctc_patch.py,gui_table_patch.py} "$APP/"
cp "$ROOT"/deploy/assets/{deploy_thresholds.json,dict.ltr.txt,f0_tone_clf.npz,priors_final.json} "$ROOT/data/vocab.csv" "$APP/"
mkdir -p "$APP/ctc_data" && cp "$APP/dict.ltr.txt" "$APP/ctc_data/"
(cd "$APP" && python3 ctc_endpoint_patch.py && python3 gui_ctc_patch.py && python3 gui_table_patch.py)
(cd "$APP/GUI" && export npm_config_cache="$ROOT/build/npm-cache" && npm ci --no-audit --no-fund >/dev/null && npm run build >/dev/null && rm -rf node_modules)
MD="$(dirname "$MODEL")"; MB="$(basename "$MODEL")"
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$OUT" -C "$APP" api_similarity_server.py freeze_layers_hook.py fairseq GUI/dist \
    ctc_scorer.py f0_infer.py f0_tone_clf.npz priors_final.json deploy_thresholds.json dict.ltr.txt ctc_data vocab.csv \
    -C "$MD" --transform "s|^${MB}\$|ctc_checkpoint.pt|" "$MB"
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1), $(tar -tf "$OUT" | wc -l) entries)"
