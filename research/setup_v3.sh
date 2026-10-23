#!/bin/bash
# 统一作业入口: setup_v3.sh <train|eval> <v1|v2|v3> <0-3|final>
#   train: 训 v3 → best ckpt 写 staging → (fold 0-3) 立即对留出折抽 loss
#   eval : 用已有 v1/v2 ckpt 对留出折抽 loss
set -e
KIND="$1"; MODEL="$2"; FOLD="$3"; ST=/staging/x/xouyang7
export INITIAL_DIR="$(pwd)"; mkdir -p joblogs
set -o pipefail
trap 'tar czf "${INITIAL_DIR}/joblogs_${KIND}_${MODEL}_${FOLD}.tar.gz" -C "${INITIAL_DIR}" joblogs 2>/dev/null || true' EXIT
echo "=== v3 batch: kind=$KIND model=$MODEL fold=$FOLD host=$(hostname) ==="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
tar -xzf fairseq.tar.gz; tar -xzf manifest_full33.tar.gz
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
if [ "$KIND" = train ]; then
  set +e; bash run_v3.sh "$FOLD" rlz; RC=$?; set -e
  echo "train exit=$RC"
  [ "$FOLD" = final ] && exit $RC
  CKPT="${INITIAL_DIR}/checkpoints_v3_fold${FOLD}_rlz/checkpoint_best.pt"
  [ -f "$CKPT" ] || { echo "无 best ckpt，跳过抽 loss"; exit 1; }
else
  case "$MODEL" in
    v1) CKPT=$ST/full32_ckpts/full32_fold${FOLD}_best.pt ;;
    v2) CKPT=$ST/full32v2_ckpts/v2_fold${FOLD}_best.pt ;;
  esac
  [ -f "$CKPT" ] || { echo "缺 $CKPT"; exit 1; }
fi
echo "=== extract losses: $CKPT ==="
OUT="${MODEL}_fold${FOLD}_loss.npz"
python extract_losses.py --ckpt "$CKPT" --w2v-path $ST/xlsr_300m.pt \
  --manifest-dir manifest_full33/fold${FOLD}_rlz --labels-csv manifest_full33/labels/fold${FOLD}_heldout.csv \
  --index full33_index.csv --data-dir . --out "$OUT" 2>&1 | grep -v -i warning | tee joblogs/extract_${MODEL}_${FOLD}.log
mkdir -p $ST/evals33 && cp "$OUT" "$ST/evals33/$OUT.part" && mv "$ST/evals33/$OUT.part" "$ST/evals33/$OUT"
ls -l "$ST/evals33/$OUT"
echo "=== done ==="; date
