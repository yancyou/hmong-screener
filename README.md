# Hmong pronunciation screener

The screener ranks a patient's recordings of a known Hmong word list so the likely mispronunciations come first. A closed-vocabulary CTC model (XLS-R 300M fine-tuned on 30 speakers) scores each recording against its target word and that word's likely confusions, and a pitch-based classifier adds a second tone check. It runs as the Batch Score page of the lab's Hmong audio web app at <https://youngxuan-hmong-screener.hf.space>.

This repository holds everything we built: data preparation, training, evaluation, the deployed scorer, the patches that add it to the web app, website deployment and tests. The web app itself is a teammate's project and is not public; lab members find the version we patch archived on staging.

## Layout

| Folder | Contents |
| --- | --- |
| `data/` | The 1,225-word list, confusion priors, checksums of the large files on staging; the label indexes and speaker folds are private (`data/README.md`) |
| `prep/` | Label parsing, audio conversion, label merge, training manifests, priors |
| `train/` | CHTC job files for fine-tuning with the deployed (v1) recipe |
| `evaluate/` | Held-out loss extraction (GPU), offline scoring and model comparison; `f0/` pitch classifier; `cosine/` comparison with the teammate's model |
| `deploy/` | Scorer, model-side assets, web-app patches, deploy-kit documents, kit and website build scripts |
| `tests/` | `verify.py` reproducibility checks and GPU smoke tests |
| `research/` | Earlier experiments kept for reference (33-speaker reruns, model soups, voice-quality features); not needed to run the system |
| `docs/` | Verification record |

## Large files

Large files live on CHTC staging under `/staging/x/xouyang7` (set `HMONG_STAGING` to move them); `data/staging_files.sha256` lists their checksums.

| File | What it is |
| --- | --- |
| `full32_16k.tar` | 42,668 clips from 32 speakers, 16 kHz mono 16-bit WAV |
| `full33_add_16k.tar` | 2,059 clips added in September 2026 (one new speaker and the rest of another speaker's recordings) |
| `hmong_segments/` | Raw downloads from Box: shards and their index |
| `xlsr_300m.pt` | XLS-R 300M base model |
| `full32_ckpts/full32_foldfinal_infer.pt` | Deployed model, an inference copy of `full32_foldfinal_best.pt`; also on the v1.0 release (see Model) |
| `full32_ckpts/full32_fold{0,2,3}_best.pt`, `full32_fold1_retrain_best.pt` | Cross-validation models. `full32_fold1_best.pt` is a mislabeled copy of fold 3; see Known issues |
| `evals33/` | Held-out loss matrices for every evaluated model |
| `hf_bundle.tar` | The website's private bundle (code and model) |
| `ctc_deploy_kit.tar.gz` | Deploy kit for lab computers |
| `upstream/hmong_audio_software-main_20261004.tar.gz` | The web-app version we patch |
| `cosine_checkpoint_slim.pt` | The teammate's similarity model |

## Private files

The recordings come from research participants, so this repository holds only code, the word list and aggregate statistics. The audio, the per-recording labels, the speaker codes and the internal notes are not public. Lab members with CHTC access restore them into the repository root:

```bash
tar -xzf /staging/x/xouyang7/hmong-screener-private.tar.gz
```

The archive holds the label indexes, folds and derived arrays in `data/`, `data/speakers.json` (speaker settings some scripts read), the 12 website test recordings, the website environment test (`tests/hf_env/`) and the internal notes in `docs/`. `.gitignore` keeps them out of git, and `tests/verify.py` fails if a tracked file contains a speaker code. Without these files, data preparation, evaluation and most checks cannot run; `verify.py` skips those checks.

## Model

The deployed model is attached to the [v1.0 release](https://github.com/yancyou/hmong-screener/releases/tag/v1.0): `full32_foldfinal_infer.pt`, 1.26 GB, SHA-256 `8279711a33676c8d7c5ab01b1bcb1b41e0a64ab679e6c2254c10c7845ea3db66`. It is XLS-R 300M fine-tuned with CTC on 30 speakers (update 6,000 of the recipe under Training). Scoring also needs the files in `deploy/assets/`, which are in this repository.

```bash
curl -L -o build/full32_foldfinal_infer.pt \
  https://github.com/yancyou/hmong-screener/releases/download/v1.0/full32_foldfinal_infer.pt
python3 deploy/scorer/deploy_screen.py --ckpt build/full32_foldfinal_infer.pt --vocab data/vocab.csv --wav a.wav --word tsheb
```

## Requirements

- A CHTC account with access to the staging folder. Run everything on the access point `ap2002.chtc.wisc.edu`.
- The access point's Python 3.9 with numpy, scipy and soundfile runs data preparation, offline evaluation and `tests/verify.py`.
- GPU jobs use the container `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` and install `train/requirements.txt` plus the bundled fairseq fork (`train/fairseq.tar.gz`). One GPU with at least 40 GB of memory and compute capability 8.0.
- The website build needs Node 20 and Python 3. On CHTC: `export PATH=/home/xouyang7/hmong/_swtest/node/bin:$PATH`.
- Deploying the website needs a Hugging Face write token in `~/.hf_token`. Never paste tokens into chats or commit them.

## Running each component

Run every command from the repository root unless it says `cd`. Scripts read from `data/` and write to `build/`, so a rerun never overwrites the canonical copies.

### 1. Data preparation

```bash
python3 prep/parse_labels.py      # raw file names -> labels before the merge, speaker folds, priors
python3 prep/build_16k.py         # raw audio -> build/full32_16k.tar and the pre-merge index
python3 prep/merge_labels.py --xlsx <Excel export>   # three-source label merge; the export is missing (Known issues)
python3 prep/build_16k_add.py     # September additions -> build/full33_add_16k.tar and build/full33_index.csv
bash prep/make_job_tarballs.sh    # manifests for training and evaluation jobs
python3 prep/make_priors.py       # leak-free confusion priors, one per fold plus the final one
```

### 2. Training

```bash
bash prep/make_job_tarballs.sh
cd train && condor_submit train.sub    # folds in fold_list.txt; "final" is the deployed model
# Retrain one fold and keep its best checkpoint on staging: edit the arguments in retrain.sub, then condor_submit retrain.sub
```

The v1 recipe: 20,000 updates; Adam, learning rate 5e-5, tri-stage schedule; 1.28 million samples per step with 4 steps per update; transformer frozen for the first 2,000 updates; fp16; best checkpoint by validation word error rate at every 2,000-update save. It takes about 3.5 hours on one NVIDIA L40.

### 3. Evaluation

```bash
cd evaluate && cp evalck_jobs.example.txt evalck_jobs.txt && condor_submit evalck.sub && cd ..   # GPU: held-out losses
python3 evaluate/compare_models.py    # metrics and speaker-level bootstrap
python3 evaluate/select_model.py      # model choice on the deployed candidate set
python3 evaluate/eval_combos.py       # ensembles and pitch fusion
```

Metric definitions are in the handover document and at the top of `evaluate/compare_models.py`.

### 4. Pitch (F0) classifier

```bash
python3 evaluate/f0/make_f0_shards.py
cd evaluate/f0 && condor_submit f0_extract.sub && cd ../..   # 8 CPU jobs
python3 evaluate/f0/train_f0_classifier.py                   # -> build/f0_tone_clf.npz
```

### 5. Deployment

```bash
bash deploy/build_kit.sh                  # lab-computer kit -> build/ctc_deploy_kit.tar.gz
bash deploy/website/build_bundle.sh       # website bundle -> build/hf_bundle.tar
python3 deploy/website/deploy_to_hf.py    # upload, restart the Space, check 12 recordings of known result
```

Installing on a lab computer: `deploy/kit_docs/INSTALL_AND_TEST.md`. Website operations: `deploy/kit_docs/DEPLOY_WEB.md`. Both are in Chinese; the handover document covers the same ground in English.

### 6. Tests

```bash
python3 tests/verify.py           # quick checks, about a minute
python3 tests/verify.py --full    # adds label re-parse, audio spot-check, bundle rebuild, evaluation reruns (about 35 minutes)
python3 tests/verify.py --live    # also checks the public website (needs the private test recordings)
cd tests/gpu_smoke && condor_submit train_smoke.sub && condor_submit eval_smoke.sub   # GPU pipelines from this repository
```

## Verification

The results of the checks run on 2026-10-23 are in `docs/VERIFICATION.md`.

## Known issues

- **The v1 fold-1 evaluation used the wrong model.** `full32_fold1_best.pt` is a copy of the fold-3 model, which was trained on fold 1's test speakers, so earlier cross-validation numbers were too high. `docs/VERIFICATION.md` gives the corrected numbers and the retrained fold-1 model.
- **The label merge cannot be rerun.** The Excel export it reads was not kept. `data/full32_merged.csv` keeps both the pre-merge `label` and `label_merged`, so every later step is reproducible.
- **The web-app snapshot is reconstructed.** The teammate's repository commit is unknown; the archived copy was rebuilt from our deployed copy, and our patches applied to it reproduce the deployed website byte for byte.
- **Personal accounts.** The CHTC files and the Hugging Face Space belong to personal accounts and should move to lab-owned ones.

## License

The code is released under the MIT License (`LICENSE`). `train/fairseq.tar.gz` is a snapshot of fairseq (MIT License, Meta Platforms; its license is inside the archive). The web app that `deploy/patches/` modifies belongs to its authors; the patch scripts quote short parts of it to find where to insert changes. No model weights, recordings or participant labels are distributed here.
