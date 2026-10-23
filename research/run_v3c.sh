#!/bin/bash
set -e
# v3b: 与 v1/v2/v3 相同超参；额外保留最后 5 个间隔 checkpoint，训完后产出 best / last / avg5 三个版本。
# 用法: run_v3c.sh <0-3|final|all> <seed>   (= run_v3b.sh + 指定随机种子)
FOLD="${1:?need fold}"; SEED="${2:?need seed}"
case "$FOLD" in final) MDIR=final_rlz;; all) MDIR=all_rlz;; *) MDIR="fold${FOLD}_rlz";; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DATA="${SCRIPT_DIR}/manifest_full33/${MDIR}"
BASE_MODEL="/staging/x/xouyang7/xlsr_300m.pt"
OUT_DIR="${SCRIPT_DIR}/checkpoints_v3b_s${SEED}_${MDIR}"
LOG_DIR="${SCRIPT_DIR}/logs_v3b_s${SEED}_${MDIR}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
DATA_ROOT="${SCRIPT_DIR}/full32_16k"
for tsv in "${ASR_DATA}"/*.tsv; do sed -i "1c${DATA_ROOT}" "$tsv"; done
miss=0
while IFS=$'\t' read -r fn _; do [ -z "$fn" ] && continue; [ -f "${DATA_ROOT}/${fn}" ] || miss=$((miss+1)); done < <(tail -n +2 "${ASR_DATA}/train.tsv" | shuf -n 50 --random-source=<(yes))
echo "spot-check 50 train wavs: missing=$miss"; [ "$miss" -gt 0 ] && { echo "audio missing"; exit 1; }
export PYTHONPATH="${SCRIPT_DIR}/fairseq:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
CONFIG_DIR="${SCRIPT_DIR}/fairseq/examples/wav2vec/xlsr/config"
LOG_FILE="${LOG_DIR}/train_${MDIR}.log"
echo "=== v3b ASR finetune ${MDIR} (train $(($(wc -l < ${ASR_DATA}/train.tsv)-1)) clips) ==="
set +e
python -m fairseq_cli.hydra_train \
  --config-dir "${CONFIG_DIR}" --config-name finetune \
  task.data="${ASR_DATA}" task.normalize=true task.labels=ltr \
  model.w2v_path="${BASE_MODEL}" model.freeze_finetune_updates=2000 \
  model.apply_mask=true model.mask_prob=0.2 +model.mask_length=3 \
  model.mask_channel_prob=0.1 model.mask_channel_length=16 \
  optimization.max_update=20000 optimization.lr='[0.00005]' optimization.update_freq='[4]' \
  dataset.max_tokens=1280000 dataset.valid_subset=valid \
  dataset.validate_after_updates=1000 dataset.validate_interval_updates=1000 \
  distributed_training.distributed_world_size=1 \
  checkpoint.save_dir="${OUT_DIR}" checkpoint.save_interval_updates=2000 \
  checkpoint.keep_interval_updates=5 checkpoint.best_checkpoint_metric=wer \
  checkpoint.no_epoch_checkpoints=true \
  common.seed=${SEED} common.fp16=true common.log_format=json common.log_interval=50 \
  common.tensorboard_logdir="${LOG_DIR}/tb" \
  hydra.run.dir="${SCRIPT_DIR}/outputs/v3b_s${SEED}_${MDIR}" \
  2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e
echo "train exit=${RC}"; ls -la "${OUT_DIR}"
[ -f "${OUT_DIR}/checkpoint_last.pt" ] || { echo "NO checkpoint_last.pt"; exit 1; }
echo "=== average last 5 interval checkpoints ==="
python "${SCRIPT_DIR}/fairseq/scripts/average_checkpoints.py" --inputs "${OUT_DIR}" \
  --num-update-checkpoints 5 --output "${OUT_DIR}/checkpoint_avg5.pt"
echo "=== strip optimizer state & copy to staging ==="
ST=/staging/x/xouyang7/full33v3_ckpts; mkdir -p "$ST"
for V in best last avg5; do
  SRC="${OUT_DIR}/checkpoint_${V}.pt"; [ -f "$SRC" ] || { echo "skip $V (missing)"; continue; }
  python - "$SRC" "$ST/v3b_s${SEED}_${FOLD}_${V}.pt" <<'PY'
import sys,os,torch
src,dst=sys.argv[1],sys.argv[2]
ck=torch.load(src,map_location='cpu')
ck.pop('last_optimizer_state',None)          # optimizer_history 必须保留, 否则 fairseq 加载时 KeyError
upd=ck.get('optimizer_history',[{}])[-1].get('num_updates')
torch.save(ck,dst+'.part'); os.replace(dst+'.part',dst)
print(f"  {os.path.basename(dst)}  updates={upd}  {os.path.getsize(dst)/2**30:.2f} GB")
PY
done
mkdir -p "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs"
cp "${LOG_FILE}" "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs/" 2>/dev/null || true
cp "${ASR_DATA}/split.json" "${INITIAL_DIR:-$SCRIPT_DIR}/joblogs/split_s${SEED}_${FOLD}.json" 2>/dev/null || true
exit "${RC}"
