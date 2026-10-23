#!/bin/bash
# Held-out CTC losses for any checkpoint: setup_evalck.sh <fold> <checkpoint path on staging> <output name>
# Writes $HMONG_EVALS_OUT/<name>_fold<k>_loss.npz (default /staging/x/xouyang7/evals33); use a new name to avoid replacing existing results.
set -e
FOLD="$1"; CKPT="$2"; NAME="$3"; ST=/staging/x/xouyang7
export INITIAL_DIR="$(pwd)"; mkdir -p joblogs
set -o pipefail
trap 'tar czf "${INITIAL_DIR}/joblogs_evalck_${NAME}_${FOLD}.tar.gz" -C "${INITIAL_DIR}" joblogs 2>/dev/null || true' EXIT
echo "=== evalck fold=$FOLD ckpt=$CKPT name=$NAME host=$(hostname) ==="; date
[ -f "$CKPT" ] || { echo "缺 $CKPT"; exit 1; }
tar -xzf fairseq.tar.gz; tar -xzf manifest_full33.tar.gz
cp $ST/full32_16k.tar . && tar -xf full32_16k.tar && rm full32_16k.tar
cp $ST/full33_add_16k.tar . && tar -xf full33_add_16k.tar && rm full33_add_16k.tar
export PATH=$HOME/.local/bin:$PATH; export PYTHONPATH="${INITIAL_DIR}/fairseq:${PYTHONPATH}"
python -m pip install --user pip==24.0 2>&1|tail -1
python -m pip install --user -r requirements.txt 2>&1|tail -2
python -m pip install --user cython "numpy<2.0" "torch==2.1.0" --no-deps soundfile librosa editdistance 2>&1|tail -1
python -m pip install --user "setuptools<75" wheel 2>&1|tail -1
cd fairseq && python -m pip install --user --no-build-isolation -e . 2>&1|tail -2 && cd ..
OUT="${NAME}_fold${FOLD}_loss.npz"
python extract_losses.py --ckpt "$CKPT" --w2v-path $ST/xlsr_300m.pt \
  --manifest-dir manifest_full33/fold${FOLD}_rlz --labels-csv manifest_full33/labels/fold${FOLD}_heldout.csv \
  --index full33_index.csv --data-dir . --out "$OUT" ${EXTRACT_ARGS:-} 2>&1 | grep -v -i warning | tee joblogs/extract.log
EV="${HMONG_EVALS_OUT:-$ST/evals33}"
mkdir -p "$EV" && cp "$OUT" "$EV/$OUT.part" && mv "$EV/$OUT.part" "$EV/$OUT"
ls -l "$EV/$OUT"; echo "=== done ==="; date
