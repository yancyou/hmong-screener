#!/bin/bash
set -e
# v3 (33 speakers) ASR finetune (XLSR-300M CTC). 用法: run_full32.sh <fold:0-3|final> [variant:rlz|l1]
FOLD="${1:?need fold}"; VARIANT="${2:-rlz}"
[ "$FOLD" = "final" ] && MDIR="final_rlz" || MDIR="fold${FOLD}_${VARIANT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DATA="${SCRIPT_DIR}/manifest_full33/${MDIR}"
BASE_MODEL="/staging/x/xouyang7/xlsr_300m.pt"
OUT_DIR="${SCRIPT_DIR}/checkpoints_v3_${MDIR}"
LOG_DIR="${SCRIPT_DIR}/logs_v3_${MDIR}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

DATA_ROOT="${SCRIPT_DIR}/full32_16k"
for tsv in "${ASR_DATA}"/*.tsv; do
  sed -i "1c${DATA_ROOT}" "$tsv"
done
miss=0
while IFS=$'\t' read -r fn _; do [ -z "$fn" ] && continue; [ -f "${DATA_ROOT}/${fn}" ] || miss=$((miss+1)); done < <(tail -n +2 "${ASR_DATA}/train.tsv" | head -5)
echo "spot-check train wavs: $((5-miss))/5"; [ "$miss" -gt 0 ] && { echo "audio missing"; exit 1; }

export PYTHONPATH="${SCRIPT_DIR}/fairseq:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
CONFIG_DIR="${SCRIPT_DIR}/fairseq/examples/wav2vec/xlsr/config"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "=== full32 ASR finetune fold=${FOLD} variant=${VARIANT} ==="

set +e
python -m fairseq_cli.hydra_train \
  --config-dir "${CONFIG_DIR}" \
  --config-name finetune \
  task.data="${ASR_DATA}" \
  task.normalize=true \
  task.labels=ltr \
  model.w2v_path="${BASE_MODEL}" \
  model.freeze_finetune_updates=2000 \
  model.apply_mask=true \
  model.mask_prob=0.2 \
  +model.mask_length=3 \
  model.mask_channel_prob=0.1 \
  model.mask_channel_length=16 \
  optimization.max_update=20000 \
  optimization.lr='[0.00005]' \
  optimization.update_freq='[4]' \
  dataset.max_tokens=1280000 \
  dataset.valid_subset=valid \
  dataset.validate_after_updates=1000 \
  dataset.validate_interval_updates=1000 \
  distributed_training.distributed_world_size=1 \
  checkpoint.save_dir="${OUT_DIR}" \
  checkpoint.save_interval_updates=2000 \
  checkpoint.keep_interval_updates=1 \
  checkpoint.best_checkpoint_metric=wer \
  checkpoint.no_epoch_checkpoints=true \
  common.fp16=true \
  common.log_format=json \
  common.log_interval=50 \
  common.tensorboard_logdir="${LOG_DIR}/tb" \
  hydra.run.dir="${SCRIPT_DIR}/outputs/v3_${MDIR}_$(date +%Y%m%d_%H%M%S)" \
  2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

echo "=== copy best checkpoint to staging ==="
ST=/staging/x/xouyang7/full33v3_ckpts; mkdir -p "$ST"
BEST="${OUT_DIR}/checkpoint_best.pt"
if [ -f "$BEST" ]; then
  cp "$BEST" "$ST/v3_${FOLD}_best.pt.part" && mv "$ST/v3_${FOLD}_best.pt.part" "$ST/v3_${FOLD}_best.pt"
  ls -l "$ST/v3_${FOLD}_best.pt"
else echo "NO checkpoint_best.pt"; fi
mkdir -p "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs"
cp "${LOG_DIR}"/*.log "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs/" 2>/dev/null || true
cp "${ASR_DATA}/split.json" "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs/split_${FOLD}.json" 2>/dev/null || true
exit "${RC}"
