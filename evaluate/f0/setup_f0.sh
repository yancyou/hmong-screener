#!/bin/bash
set -e
export INITIAL_DIR="$(pwd)"; echo "=== F0 feature extraction, shard=$1 ==="; date
cp /staging/x/xouyang7/full32_16k.tar . && tar -xf full32_16k.tar && rm full32_16k.tar
export PATH=$HOME/.local/bin:$PATH
python -m pip install --user pip==24.0 2>&1|tail -1
python -m pip install --user "numpy<2.0" soundfile librosa 2>&1|tail -1
python -u extract_f0_features.py --labels-csv f0_shard$1.csv --data-dir . --out f0_shard$1.npz --workers 16
echo "=== done ==="; date; ls -lh f0_shard$1.npz
