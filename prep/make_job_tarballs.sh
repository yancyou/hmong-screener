#!/bin/bash
# Build the manifest tarballs the CHTC jobs transfer (run from anywhere):
#   train/manifest_full32.tar.gz    v1 manifests (32 speakers, pre-merge labels), the deployed model's training data
#   evaluate/manifest_full33.tar.gz 33-speaker manifests (4 folds + final + all), used for held-out evaluation
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
rm -rf build/manifest_v1 build/manifest_full33
python3 prep/make_manifests.py --index data/full32_merged.csv --folds data/folds.json --out build/manifest_v1 >/dev/null
tar -C build -czf train/manifest_full32.tar.gz --transform 's|^manifest_v1|manifest_full32|' manifest_v1
python3 prep/make_manifests.py --index data/full33_index.csv --folds data/folds33.json --dict deploy/assets/dict.ltr.txt --out build/manifest_full33 >/dev/null
python3 prep/make_all_manifest.py >/dev/null
tar -C build -czf evaluate/manifest_full33.tar.gz manifest_full33
ls -l train/manifest_full32.tar.gz evaluate/manifest_full33.tar.gz | awk '{print $5, $9}'
