#!/bin/bash
# setup_v3c.sh <0-3|final|all> <seed>
#   训 v3b → best/last/avg5 三个版本写 staging → (fold 0-3) 对留出折逐个抽 loss
set -e
FOLD="$1"; SEED="$2"; ST=/staging/x/xouyang7
export INITIAL_DIR="$(pwd)"; mkdir -p joblogs
set -o pipefail
trap 'tar czf "${INITIAL_DIR}/joblogs_v3c_s${SEED}_${FOLD}.tar.gz" -C "${INITIAL_DIR}" joblogs 2>/dev/null || true' EXIT
echo "=== v3c fold=$FOLD seed=$SEED host=$(hostname) ==="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
tar -xzf fairseq.tar.gz; tar -xzf manifest_full33b.tar.gz
cp $ST/full32_16k.tar . && tar -xf full32_16k.tar && rm full32_16k.tar
cp $ST/full33_add_16k.tar . && tar -xf full33_add_16k.tar && rm full33_add_16k.tar
NW=$(find full32_16k -name '*.wav' | wc -l); echo "wavs: $NW"; [ "$NW" -ge 44700 ] || { echo "音频不全"; exit 1; }
export PATH=$HOME/.local/bin:$PATH; export PYTHONPATH="${INITIAL_DIR}/fairseq:${PYTHONPATH}"
python -m pip install --user pip==24.0 2>&1|tail -1
python -m pip install --user -r requirements.txt 2>&1|tail -2
python -m pip install --user cython "numpy<2.0" "torch==2.1.0" --no-deps soundfile librosa editdistance 2>&1|tail -1
python -m pip install --user "setuptools<75" wheel 2>&1|tail -1
cd fairseq && python -m pip install --user --no-build-isolation -e . 2>&1|tail -2 && cd ..
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
bash run_v3c.sh "$FOLD" "$SEED"
case "$FOLD" in final|all) echo "=== done (部署候选, 无留出折) ==="; date; exit 0;; esac
mkdir -p $ST/evals33
for V in best last avg5; do
  CK="${INITIAL_DIR}/checkpoints_v3b_s${SEED}_fold${FOLD}_rlz/checkpoint_${V}.pt"
  [ -f "$CK" ] || { echo "缺 $CK"; continue; }
  OUT="v3bs${SEED}${V}_fold${FOLD}_loss.npz"
  echo "=== extract losses: $V ==="
  python extract_losses.py --ckpt "$CK" --w2v-path $ST/xlsr_300m.pt \
    --manifest-dir manifest_full33/fold${FOLD}_rlz --labels-csv manifest_full33/labels/fold${FOLD}_heldout.csv \
    --index full33_index.csv --data-dir . --out "$OUT" 2>&1 | grep -v -i warning | tee joblogs/extract_s${SEED}_${V}_${FOLD}.log
  cp "$OUT" "$ST/evals33/$OUT.part" && mv "$ST/evals33/$OUT.part" "$ST/evals33/$OUT"
  ls -l "$ST/evals33/$OUT"
done
echo "=== done ==="; date
