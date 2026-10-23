# Research scripts

Earlier experiments, kept as they were run for reference. None is needed to train, evaluate or deploy the screener.

| Files | Experiment |
| --- | --- |
| `setup_v3*.sh`, `run_v3*.sh`, `v3*.sub` | 33-speaker reruns (two seeds; best, last and 5-checkpoint average). None beat v1. |
| `soup.py` | Weight averaging of models fine-tuned with different seeds. |
| `closed_vocab_verify*.py` | First proof of concept of closed-vocabulary rescoring. |
| `eval_three_way.py`, `extract_vq_features.py` | Adding voice-quality features to the tone decision (dropped). |
| `aggregate_full32.py` | Early 32-speaker cross-validation report. |
| `demo_ctc.py` | Prints the CTC scoring of one recording step by step. |
| `collect_*.sh` | Collected job outputs onto staging. `collect_full32.sh` is where the fold-1 / fold-3 mix-up most likely happened. |
