#!/bin/bash
# Train one fold with the v1 recipe and copy checkpoint_best.pt plus the training log to staging, instead of
# returning the 7 GB output tarball to the access point.  Usage (inside a job): train_to_staging.sh <fold|final> <dest.pt>
set -e
FOLD="$1"; DEST="$2"
bash setup_train.sh "$FOLD" rlz
[ "$FOLD" = "final" ] && MDIR="final_rlz" || MDIR="fold${FOLD}_rlz"
cp "checkpoints_full32_${MDIR}/checkpoint_best.pt" "${DEST}.part" && mv "${DEST}.part" "${DEST}"
cp "$(ls -t logs_full32_${MDIR}/*.log | head -1)" "${DEST%.pt}.log"
rm -f "full32_${FOLD}_rlz_output.tar.gz"
echo "saved ${DEST}" | tee retrain_result.txt
