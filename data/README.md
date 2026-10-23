# Data files

Small metadata only; the audio and checkpoints are on CHTC staging (`staging_files.sha256`). Files marked *private* hold participant-level data and are not in the public repository; lab members restore them from the private archive (top-level README, "Private files").

| File | What it is |
| --- | --- |
| `full33_index.csv` (private) | Current labels, 33 speakers, 44,727 rows. `label` 1 correct, 2 tone error, 3 whole-word error, 0 uncertain; `realized` is what was actually said; `suspect` 1 means the note contradicts the target and the clip is not used as a training transcript; `src` is label reliability (`agree-ok`, `agree-err`, `cur-only`, `union-err` = single source, `cur` = uncertain). Built by `prep/build_16k_add.py`. |
| `full32_merged.csv` (private) | 32 speakers with both the pre-merge `label` (used to train the deployed v1 model) and `label_merged` plus `src`. Built by `prep/merge_labels.py`. |
| `full32_index.csv` (private) | v2 training index: `label` = merged label, new errors without a transcript flagged `suspect`. Built by `prep/merge_labels.py`. |
| `folds.json` (private) | 4-fold speaker split for the 32 speakers (8 test speakers per fold). Built by `prep/parse_labels.py`. |
| `folds33.json` (private) | The same folds with the September speaker added to fold 2. |
| `split_foldfinal.json` (private) | The deployed model's split: 30 training speakers, 2 validation speakers, 38,935 / 2,566 clips. |
| `confusion_priors.json` | Confusions seen across all 32 speakers (leaky; kept for the leakage comparison). |
| `priors33/` | Confusion priors from each fold's training speakers only (`priors_fold{k}.json`) and from all 33 (`priors_final.json`, deployed). |
| `vocab.csv` | The 1,225-word study list the website scores against. |
| `sources/new16k_index.csv` (private) | The June 2026 labeling pass, one of the three merge sources. |
| `derived/` (private) | Saved pitch-fusion inputs (`P_f0_full32.npy`, `f0_full32_meta.npz`, `ctc_by_wav.json`) and the label list used to shard pitch extraction. |
| `staging_files.sha256` | Checksums and sizes of the large files on staging. |
| `speakers.json` (private) | Speaker settings scripts read: the two speakers pinned to fold 0 (`prep/parse_labels.py`) and the validation speakers of the all-speaker run (`prep/make_all_manifest.py`). |
