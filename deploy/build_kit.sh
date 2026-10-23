#!/bin/bash
# Build the deploy kit (ctc_deploy_kit.tar.gz): everything a lab computer needs to add the screening mode
# to the Hmong audio app, except the 1.2 GB model file. Run from anywhere; output defaults to build/.
#   bash deploy/build_kit.sh [output.tar.gz]
# Needs two private files (README, "Private files"): deploy/kit_docs/AUDIT.md and data/full33_index.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/ctc_deploy_kit.tar.gz}"
STAGE="$ROOT/build/kit_stage"; rm -rf "$STAGE"; mkdir -p "$STAGE" "$(dirname "$OUT")"
cp "$ROOT"/deploy/kit_docs/{AUDIT.md,CTC_DEPLOY.md,DEPLOY_WEB.md,INSTALL_AND_TEST.md} "$STAGE/"
cp "$ROOT"/deploy/patches/{ctc_endpoint_patch.py,gui_ctc_patch.py,gui_table_patch.py,web_deploy_patch.py} "$STAGE/"
cp "$ROOT"/deploy/scorer/{ctc_scorer.py,f0_infer.py} "$STAGE/"
cp "$ROOT"/deploy/assets/{deploy_thresholds.json,dict.ltr.txt,f0_tone_clf.npz,priors_final.json} "$STAGE/"
cp "$ROOT"/data/full33_index.csv "$STAGE/"        # word list for local installs (the website uses data/vocab.csv)
tar -C "$STAGE" -czf "$OUT" AUDIT.md CTC_DEPLOY.md ctc_endpoint_patch.py ctc_scorer.py deploy_thresholds.json DEPLOY_WEB.md \
    dict.ltr.txt f0_infer.py f0_tone_clf.npz full33_index.csv gui_ctc_patch.py gui_table_patch.py INSTALL_AND_TEST.md \
    priors_final.json web_deploy_patch.py
echo "kit: $OUT ($(du -h "$OUT" | cut -f1), $(tar -tzf "$OUT" | wc -l) files)"
