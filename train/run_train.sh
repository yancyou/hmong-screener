#!/bin/bash
set -e
# XLS-R 300M CTC fine-tuning, the recipe of the deployed v1 model. Usage: run_train.sh <fold:0-3|final> [variant:rlz|l1]
FOLD="${1:?need fold}"; VARIANT="${2:-rlz}"
# MAX_UPDATE / VALID_INTERVAL / SAVE_INTERVAL override the recipe only for smoke tests (defaults = v1: 20000 / 1000 / 2000)
[ "$FOLD" = "final" ] && MDIR="final_rlz" || MDIR="fold${FOLD}_${VARIANT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DATA="${SCRIPT_DIR}/manifest_full32/${MDIR}"
BASE_MODEL="/staging/x/xouyang7/xlsr_300m.pt"
OUT_DIR="${SCRIPT_DIR}/checkpoints_full32_${MDIR}"
LOG_DIR="${SCRIPT_DIR}/logs_full32_${MDIR}"
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
  optimization.max_update=${MAX_UPDATE:-20000} \
  optimization.lr='[0.00005]' \
  optimization.update_freq='[4]' \
  dataset.max_tokens=1280000 \
  dataset.valid_subset=valid \
  dataset.validate_after_updates=${VALID_INTERVAL:-1000} \
  dataset.validate_interval_updates=${VALID_INTERVAL:-1000} \
  distributed_training.distributed_world_size=1 \
  checkpoint.save_dir="${OUT_DIR}" \
  checkpoint.save_interval_updates=${SAVE_INTERVAL:-2000} \
  checkpoint.keep_interval_updates=1 \
  checkpoint.best_checkpoint_metric=wer \
  checkpoint.no_epoch_checkpoints=true \
  common.fp16=true \
  common.log_format=json \
  common.log_interval=50 \
  common.tensorboard_logdir="${LOG_DIR}/tb" \
  hydra.run.dir="${SCRIPT_DIR}/outputs/full32_${MDIR}_$(date +%Y%m%d_%H%M%S)" \
  2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

echo "=== packing ==="
TARBALL="${INITIAL_DIR:-$SCRIPT_DIR}/full32_${FOLD}_${VARIANT}_output.tar.gz"
PACK=()
[ -f "${OUT_DIR}/checkpoint_best.pt" ] && PACK+=("${OUT_DIR}/checkpoint_best.pt")
[ -f "${OUT_DIR}/checkpoint_last.pt" ] && PACK+=("${OUT_DIR}/checkpoint_last.pt")
LATEST=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1); [ -n "$LATEST" ] && PACK+=("$LATEST")
PACK+=("${ASR_DATA}/split.json")
tar -czf "${TARBALL}" "${PACK[@]}" 2>/dev/null || echo "pack warn"
echo "tarball: $(du -h "${TARBALL}" 2>/dev/null)"
exit "${RC}"
