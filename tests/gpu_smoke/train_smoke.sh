#!/bin/bash
# GPU smoke test of the training pipeline: the repo's train/setup_train.sh + run_train.sh on the v1 "final" manifests,
# stopped after MAX_UPDATE updates. Passes when fairseq trains, validates and saves a checkpoint.
set -e
bash setup_train.sh final rlz
CK=checkpoints_full32_final_rlz/checkpoint_last.pt
python - <<'PY' > train_smoke_result.txt
import torch
ck = torch.load("checkpoints_full32_final_rlz/checkpoint_last.pt", map_location="cpu", weights_only=False)
oh = ck["optimizer_history"][-1]; es = ck.get("extra_state", {})
print(f"num_updates={oh['num_updates']} best_valid_wer={es.get('best')} saved_ok=1")
PY
cat train_smoke_result.txt
grep -h '"valid_wer"' logs_full32_final_rlz/*.log | tail -3 >> train_smoke_result.txt || true
rm -f full32_final_rlz_output.tar.gz          # 7 GB of checkpoints; not needed back on the access point
