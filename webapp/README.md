# Hmong Audio Similarity Software

Web GUI + API server for scoring how similar a candidate pronunciation is to a
reference recording of a Hmong word. Embeddings come from a fine-tuned
XLS-R (wav2vec 2.0) model; the score is the raw **cosine similarity ∈ [−1, 1]**
between mean-pooled embeddings of the two clips.

## Layout

```
api_similarity_server.py   FastAPI backend (loads the checkpoint, computes similarity)
run_similarity_server.sh   One-command server launcher
api_requirements.txt       Python dependencies for the backend
fairseq/                   Vendored fairseq with our custom tasks (gender_tone_repulsive, ...)
GUI/                       React + Vite frontend
```

Checkpoints and audio data are **not** in this repo (too large) — download
them separately, see [Required assets](#required-assets). The original XLS-R
pretraining file (`xlsr_300m.pt`) is **not** required — the server
reconstructs the model architecture from the checkpoint's own weights.

## Required assets

The model is public at https://huggingface.co/tianyi0216/hmong-xlsr-checkpoint (checkpoint_slim.pt).
The reference recordings come from research participants and are not public; ask the lab.

| Asset | File | Where to put it |
|---|---|---|
| Fine-tuned model | `checkpoint_last_1.pt` (~2.4 GB) | anywhere; pass its path to `run_similarity_server.sh` |
| Reference word recordings | `reference_words.zip` (~77 MB, 5400 wavs + file list) | **unzip into this repo's root** so `all_data/` sits next to `api_similarity_server.py` — this enables the GUI's word-library dropdown (1143 words, via `template_manifest_all_sources_with_xlsx.csv`, which comes with the recordings) |

The reference set contains standard pronunciations of the Hmong word lists
(two speakers per word). In the GUI, load a reference word clip on the left,
then record or upload the candidate pronunciation on the right.

## Environment setup (once)

Requires: Python ≥ 3.9 (conda recommended), Node.js ≥ 18.

```bash
# 1. Python env
conda create -n hmong-env python=3.10 -y
conda activate hmong-env
pip install -r api_requirements.txt
pip install -e ./fairseq          # vendored fairseq with custom tasks

# 2. Frontend deps
cd GUI && npm install && cd ..
```

On a machine that already has the `hmong-env` conda env (e.g. the lab Mac),
skip step 1 and just `conda activate hmong-env`.

## Run

```bash
# Terminal 1 — backend (port 8000)
conda activate hmong-env
./run_similarity_server.sh /path/to/checkpoint_last_1.pt 20   # <ckpt> [layer]

# Terminal 2 — frontend (port 5173)
cd GUI && npm run dev
```

Open http://localhost:5173. The model loads lazily on the first request
(~2–3 min for a 2.4 GB checkpoint); opening the GUI triggers warm-up
automatically via `/api/model_info`. `curl http://127.0.0.1:8000/api/model_info`
returns `{"num_layers":24,...}` once ready.

The GUI's **Embedding layer** box selects which transformer layer the
embedding is taken from (pre-filled with the server default; empty = final
encoder output). The Vite dev server proxies `/api` to `127.0.0.1:8000`
(see `GUI/vite.config.js`).

## API

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Loads model, returns `{"status":"ok"}` |
| `/api/model_info` | GET | `{num_layers, default_layer, device}` |
| `/api/words` | GET | Reference library: `{words:[{word, n_templates}...]}` |
| `/api/word_audio` | GET | `?word=dhia&i=0` — serve a template recording (preview) |
| `/api/similarity` | POST | multipart form: `candidate` wav(s) + `reference` wav(s) and/or `reference_word`, optional `layer` spec |

**Reference word library**: in the GUI, type in the search box on the
Reference card to pick one of the 1143 library words — all of its correct
recordings (label==1 in the manifest) become templates automatically, and
per-word prototypes are cached server-side after the first request. Uploaded
or recorded reference clips can be combined with a library word (they are
pooled into the same prototype). Set `REFERENCE_MANIFEST` /
`REFERENCE_AUDIO_ROOT` env vars to use a different library.

**Multi-template**: upload several `reference` files (repeat the field) — each
clip is embedded separately and averaged into a prototype (same for
`candidate`). **Multi-layer fusion**: `layer` accepts a single layer (`20`) or
a comma list (`19,20,21`); per-layer embeddings are L2-normalized, averaged,
and re-normalized, matching `eval/predict_pron_correctness_multitemplate.py`.

```bash
curl -F reference=@template1.wav -F reference=@template2.wav \
     -F candidate=@attempt.wav -F layer=19,20,21 \
     http://127.0.0.1:8000/api/similarity
# → {"score":0.87,"cosine":0.87,"n_reference_templates":2,"n_candidate_templates":1,
#    "explain":"emb_dim=1024 layers=19,20,21 ref_templates=2 cand_templates=1 device=mps"}
```

`score` **is** the raw cosine similarity (no remapping). In the GUI, add
multiple reference segments to average them into a template prototype, and
type `19,20,21` in the Embedding layer box for fused layers.

## Interpreting scores (layer 20, cross-speaker, measured 2026-07)

| Pair type | cosine range | mean |
|---|---|---|
| Same word | 0.78 – 0.96 | 0.89 |
| Different word | 0.29 – 0.65 | 0.46 |

The two distributions did not overlap in our sample; the GUI's default
decision threshold **0.70** separates them. Any two speech clips share
structure (voice, recording conditions), so "unrelated" cosine sits around
0.3–0.65 rather than 0 — judge scores against these bands, not against 0.

## Environment variables (backend)

| Var | Default | Meaning |
|---|---|---|
| `CHECKPOINT_PATH` | (required) | Fine-tuned `.pt` checkpoint |
| `EMBEDDING_LAYER` | `20` (via run script) | Default transformer layer, 1-based; `-1` = final output |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Server bind |
| `W2V_PATH` | unset | Optional real pretraining ckpt; unset → architecture synthesized from checkpoint |
| `DATA_DIR` | auto fallback | Dir with `dict.ltr.txt`; a minimal one is generated if unset |

## Notes

- Checkpoints trained with custom tasks (e.g. `gender_tone_repulsive`) load
  fine: if the task isn't registered, the server falls back to
  `audio_finetuning` and rebuilds the backbone from synthesized config —
  watch the log for `Loaded N/N tensors from checkpoint` to confirm full
  weight coverage.
- Input audio: WAV, any sample rate (resampled to 16 kHz, mono, layer-norm
  normalized to match training).
- Runs on CUDA / Apple MPS / CPU automatically.
