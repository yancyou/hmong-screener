# Verification, 2026-10-23

Every component was rerun from this repository and compared with the files actually used: the deployed model's training data, the stored evaluation results, the deploy kit on staging and the live website's bundle. `tests/verify.py` repeats the CPU checks; the GPU checks are in `tests/gpu_smoke/`.

## Results

| Check | How | Result |
| --- | --- | --- |
| Code health | Compile all 40 Python files, syntax-check 20 shell scripts, parse the server code the patch inserts, scan for tokens and keys | All pass; no secrets. pyflakes finds only unused imports. |
| Label parsing | `prep/parse_labels.py` on the raw Box file names, compared with the labels the deployed model used | 42,668 clips, 0 differences; speaker folds and priors identical |
| Audio conversion | `prep/build_16k.py` (recovered from the session log) on 150 raw files from shard 3, compared with `full32_16k.tar` | 150 of 150 files byte-identical |
| Deployed model's training data | `prep/make_manifests.py` with the pre-merge labels | Same 30 training speakers, 38,935 / 2,566 clips and 2,551 realized transcripts as `data/split_foldfinal.json`; letter dictionary identical to the model's |
| Later manifests and priors | Regenerated v2 and 33-speaker manifests and the per-fold priors | 247 of 247 files identical to the originals |
| Training pipeline (GPU) | `tests/gpu_smoke/train_smoke.sub`: the repository's training scripts in a fresh container, 300 updates | Trains, validates and saves a checkpoint (job 6982322, gpu2011) |
| Evaluation pipeline (GPU) | `tests/gpu_smoke/eval_smoke.sub`: held-out losses of the fold-0 model for 1,500 clips | Identical to the stored losses (largest relative difference 0); job 6982323, dsigpu4000 |
| Offline evaluation | Rerun `evaluate/select_model.py` and `evaluate/eval_combos.py` | Outputs identical to the stored `select_model.json` and `eval_combos.json` |
| Deploy kit | `deploy/build_kit.sh` | 15 of 15 files identical to `/staging/x/xouyang7/ctc_deploy_kit.tar.gz` |
| Website bundle | `deploy/website/build_bundle.sh`: archived web app + our patches + rebuilt pages + model | 1,536 of 1,536 files identical to the deployed `hf_bundle.tar` |
| Live website | 12 recordings with known results (`deploy/website/smoke/live_check.py`) | Matches the verified scores (largest difference 0.00001, no decision changed) |
| Pitch classifier | Re-extract pitch features (8 CPU jobs, cluster 6982424) and retrain with `evaluate/f0/train_f0_classifier.py` | Pending |

## Problem found: the v1 fold-1 evaluation used the wrong model

`/staging/x/xouyang7/full32_ckpts/full32_fold1_best.pt` and `full32_fold3_best.pt` are the same file (same SHA-256). Both are the fold-3 model: the configuration stored inside says it trained on `fold3_rlz`. The real fold-1 model is gone. The stored fold-1 result (`evals33/v1_fold1_loss.npz`) was computed with that file, and all 8 fold-1 test speakers were in the fold-3 model's training data, so a quarter of v1's cross-validation clips were scored by a model that had heard those speakers. The v2 and 33-speaker models were checked and are correct.

| Model | Fold 1: word / tone AUC | Folds 0, 2, 3 only: word AUC, word errors caught at 10% false alarms, tone AUC, top-1 syllable |
| --- | --- | --- |
| v1 (deployed) | 0.965 / 0.980 | 0.904, 79.2%, 0.833, 91.3% |
| v2 | 0.923 / 0.940 | 0.887, 74.6%, 0.790, 90.9% |
| v3, seed 1 best checkpoint (best of the six variants) | 0.905 / 0.946 | 0.896, 77.3%, 0.828, 91.3% |

v1's fold-1 scores stand far above every other model's, which is what in-sample scoring looks like. Without fold 1, v1 is still the best model, so the choice of deployed model stands. The previously reported cross-validation figures (word AUC 0.918, 83.0% caught, tone AUC 0.880, top-1 92.8%) were too high, and the automatic-pass threshold was calibrated on scores that include fold 1.

Fix in progress: fold 1 is being retrained with the same recipe (cluster 6986566, output `full32_ckpts/full32_fold1_retrain_best.pt`) and will be re-evaluated to give clean 4-fold numbers.
