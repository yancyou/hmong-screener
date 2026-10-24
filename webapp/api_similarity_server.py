#!/usr/bin/env python3
"""
Minimal API server for audio similarity using XLS-R/Wav2Vec CTC checkpoints.

Endpoints:
- GET /healthz                 → {status: ok}
- POST /api/similarity         → multipart/form-data with 'reference' and 'candidate' WAV files
                                 returns JSON; 'score' is the raw cosine similarity in [-1, 1]

Environment/CLI overrides:
- CHECKPOINT_PATH   (required)
- DATA_DIR          (directory containing dict.ltr.txt used during training)
- EMBEDDING_LAYER   layer spec: "" or "-1" = final encoder output; "20" = one
                    layer; "19,20,21" = multi-layer fusion (1-based indices)

The /api/similarity endpoint also accepts an optional 'layer' form field with
the same spec format, overriding EMBEDDING_LAYER per request. Multiple
'reference' (and 'candidate') files may be uploaded: each is embedded
separately and averaged into a prototype (multi-template scoring).
"""

from __future__ import annotations

import io
import os
import sys
import base64
import inspect
import secrets
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# Prefer the vendored Fairseq in ./fairseq if available
_HERE = Path(__file__).resolve().parent
_FAIRSEQ_REPO = _HERE / "fairseq"
if (_FAIRSEQ_REPO / "fairseq").exists():
    sys.path.insert(0, str(_FAIRSEQ_REPO))

# Fairseq
from fairseq import checkpoint_utils
from fairseq import tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


class Health(BaseModel):
    status: str


app = FastAPI(title="Hmong Audio Similarity API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Access control ---------------------------------------------------------
# Set ACCESS_PASSWORD to require a login before anything (page or API) can be
# used; ACCESS_USER defaults to "hmong". Leave ACCESS_PASSWORD unset and the
# server stays open, which is what local runs want.
# /healthz stays open so the hosting platform's health check keeps working.
ACCESS_USER = os.environ.get("ACCESS_USER", "hmong")
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "")
_OPEN_PATHS = {"/healthz"}


@app.middleware("http")
async def _require_login(request, call_next):
    if not ACCESS_PASSWORD or request.url.path in _OPEN_PATHS:
        return await call_next(request)

    ok = False
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            # compare_digest on both so a wrong user doesn't short-circuit
            ok = secrets.compare_digest(user, ACCESS_USER) & secrets.compare_digest(pwd, ACCESS_PASSWORD)
        except Exception:
            ok = False

    if not ok:
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Hmong Audio Tool"'},
        )
    return await call_next(request)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
MODEL = None
TASK = None
NORMALIZE = False  # set from checkpoint task cfg at load time
# Layer spec: "" or "-1" = final encoder output; "20" = single layer;
# "19,20,21" = multi-layer fusion (per-layer L2-normalized mean-pooled
# embeddings averaged, then re-normalized). 1-based indices.
EMBEDDING_LAYERS = os.environ.get("EMBEDDING_LAYER", "").strip()
W2V_PATH = os.environ.get("W2V_PATH", "")  # empty string disables loading pretrain inside checkpoint
# Reference word library: a CSV manifest with columns wav_file,word,label
# (label==1 rows are used as templates). Paths are resolved against
# REFERENCE_AUDIO_ROOT (default: the manifest's directory).
REFERENCE_MANIFEST = os.environ.get("REFERENCE_MANIFEST", "").strip()
REFERENCE_AUDIO_ROOT = os.environ.get("REFERENCE_AUDIO_ROOT", "").strip()
# Which sources may act as reference recordings. The manifest also lists
# recordings collected from study participants; scoring a participant against
# other participants is not what the reference library is for, so by default
# only the original standardized word-list recordings (all_data/) are used.
# Set REFERENCE_SOURCES to a comma-separated list of path prefixes to change
# this, or to "*" to use every source in the manifest.
REFERENCE_SOURCES = os.environ.get("REFERENCE_SOURCES", "all_data").strip()
WORD_TEMPLATES = None  # {word: [abs wav paths]}

# Leading/trailing silence is trimmed before embedding; see _trim_silence.
TRIM_SILENCE = os.environ.get("TRIM_SILENCE", "1").strip().lower() not in ("0", "false", "no")
TRIM_DB = float(os.environ.get("TRIM_DB", "-40"))
# No padding: measured on reference recordings, re-adding 25-50 ms of silence
# around the word costs 0.06-0.17 of cosine similarity, and the frame-aligned
# cut already leaves up to one 10 ms frame of lead-in, so onsets are not clipped.
TRIM_PAD_MS = float(os.environ.get("TRIM_PAD_MS", "0"))
PROTO_CACHE = {}  # {(word, layers_key): [embedding per template]}


def _synthesize_w2v_args(ckpt_state, data_dir: str, normalize: bool):
    """Reconstruct the wav2vec2 pretraining config purely from checkpoint weight
    shapes, so the server can run without the original pretraining file
    (e.g. xlsr_300m.pt). Conv strides are not stored in weights; the standard
    wav2vec2 schedule (5,2,2,2,2,2,2) is assumed."""
    import re
    from omegaconf import OmegaConf

    # Strip the "…w2v_model." prefix if this is a fine-tuned (Wav2VecCtc) checkpoint
    prefix = ""
    for k in ckpt_state:
        if ".w2v_model." in k:
            prefix = k.split(".w2v_model.")[0] + ".w2v_model."
            break
    weights = {k[len(prefix):]: v for k, v in ckpt_state.items() if k.startswith(prefix)}

    layer_ids = set()
    for k in weights:
        m = re.match(r"encoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise RuntimeError("Cannot synthesize w2v_args: no encoder layers found in checkpoint")
    n_layers = max(layer_ids) + 1
    embed_dim = int(weights["encoder.layers.0.self_attn.k_proj.weight"].shape[1])
    ffn_dim = int(weights["encoder.layers.0.fc1.weight"].shape[0])
    heads = max(1, embed_dim // 64)

    conv_layers = []
    default_strides = [5, 2, 2, 2, 2, 2, 2]
    i = 0
    while f"feature_extractor.conv_layers.{i}.0.weight" in weights:
        w = weights[f"feature_extractor.conv_layers.{i}.0.weight"]
        out_ch, _, kernel = w.shape
        stride = default_strides[i] if i < len(default_strides) else 2
        conv_layers.append((int(out_ch), int(kernel), stride))
        i += 1
    conv_bias = "feature_extractor.conv_layers.0.0.bias" in weights
    # Per-conv-layer norms → "layer_norm" extractor mode (XLS-R style), which
    # pairs with layer_norm_first transformers
    extractor_mode = "layer_norm" if "feature_extractor.conv_layers.1.2.1.weight" in weights else "default"

    return OmegaConf.create({
        "task": {"_name": "audio_pretraining", "data": data_dir, "normalize": bool(normalize)},
        "model": {
            "_name": "wav2vec2",
            "extractor_mode": extractor_mode,
            "encoder_layers": n_layers,
            "encoder_embed_dim": embed_dim,
            "encoder_ffn_embed_dim": ffn_dim,
            "encoder_attention_heads": heads,
            "activation_fn": "gelu",
            "layer_norm_first": extractor_mode == "layer_norm",
            "conv_feature_layers": str(conv_layers),
            "conv_bias": conv_bias,
            "encoder_layerdrop": 0.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "dropout_input": 0.0,
            "dropout_features": 0.0,
            "quantize_targets": False,
        },
        "criterion": None,
        "lr_scheduler": None,
    })


_LOAD_LOCK = threading.Lock()


def _load_model() -> None:
    global MODEL, TASK, NORMALIZE
    if MODEL is not None:
        return
    with _LOAD_LOCK:
        _load_model_locked()


def _load_model_locked() -> None:
    global MODEL, TASK, NORMALIZE
    if MODEL is not None:  # another request finished the load while we waited
        return
    ckpt = os.environ.get("CHECKPOINT_PATH")
    data_dir = os.environ.get("DATA_DIR")
    if not ckpt:
        raise RuntimeError("CHECKPOINT_PATH env var is required")

    # Override paths embedded in the fine-tuned checkpoint to avoid stale absolute paths
    arg_overrides = {}
    # If DATA_DIR not provided, create a minimal fallback dict locally
    if not data_dir:
        fallback = _HERE / "_fallback_data"
        fallback.mkdir(exist_ok=True)
        dict_path = fallback / "dict.ltr.txt"
        if not dict_path.exists():
            # Minimal LTR dictionary: a-z, apostrophe and word-separator '|'
            tokens = list("abcdefghijklmnopqrstuvwxyz") + ["'", "|"]
            dict_path.write_text("\n".join(f"{t} 1" for t in tokens) + "\n", encoding="utf-8")
        data_dir = str(fallback)
    if data_dir:
        arg_overrides["data"] = data_dir
    # Ensure a valid w2v_path: prefer env W2V_PATH if provided and exists; else fall back to ckpt
    try:
        w2v_override = os.environ.get("W2V_PATH", "").strip()
    except Exception:
        w2v_override = ""
    if w2v_override and Path(w2v_override).exists():
        arg_overrides["w2v_path"] = w2v_override
    else:
        arg_overrides["w2v_path"] = ckpt

    # Load the checkpoint state to CPU first
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt, arg_overrides)
    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    elif "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    else:
        raise RuntimeError("Checkpoint missing cfg/args")

    # Force override of w2v_path inside cfg to avoid stale absolute paths
    try:
        # Prefer env W2V_PATH if provided & exists; else use finetune ckpt
        w2v_override = os.environ.get("W2V_PATH", "").strip()
    except Exception:
        w2v_override = ""
    if not (w2v_override and Path(w2v_override).exists()):
        w2v_override = ckpt
    # Force all possible fields
    for path_expr in [
        ("model", "w2v_path"),
        (None, "w2v_path"),
    ]:
        try:
            parent, attr = path_expr
            if parent is None and hasattr(cfg, attr):
                setattr(cfg, attr, w2v_override)
            elif parent is not None and hasattr(cfg, parent) and hasattr(getattr(cfg, parent), attr):
                setattr(getattr(cfg, parent), attr, w2v_override)
        except Exception:
            pass
    # Disable any attempt to reload inner pretrain
    try:
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'no_pretrained_weights'):
            cfg.model.no_pretrained_weights = True
    except Exception:
        pass
    # Decide how the inner wav2vec2 backbone gets its architecture config:
    #   1. valid W2V_PATH env  → load that pretraining checkpoint (w2v_args=None forces it)
    #   2. w2v_args embedded in the fine-tuned checkpoint → use them as-is
    #   3. neither → synthesize w2v_args from the checkpoint's own weight shapes,
    #      so no external pretraining file (e.g. xlsr_300m.pt) is needed at all.
    have_env_w2v = bool(w2v_override) and w2v_override != ckpt
    orig_w2v_args = None
    try:
        if hasattr(cfg, "model") and getattr(cfg.model, "w2v_args", None) is not None:
            orig_w2v_args = cfg.model.w2v_args
    except Exception:
        orig_w2v_args = None
    if have_env_w2v:
        for path_expr in [
            ("model", "w2v_args"),
            (None, "w2v_args"),
        ]:
            try:
                parent, attr = path_expr
                if parent is None and hasattr(cfg, attr):
                    setattr(cfg, attr, None)
                elif parent is not None and hasattr(cfg, parent) and hasattr(getattr(cfg, parent), attr):
                    setattr(getattr(cfg, parent), attr, None)
            except Exception:
                pass
    elif orig_w2v_args is None:
        synth = _synthesize_w2v_args(state["model"], data_dir, getattr(cfg.model, "normalize", True))
        try:
            cfg.model.w2v_args = synth
        except Exception:
            from omegaconf import open_dict
            with open_dict(cfg.model):
                cfg.model.w2v_args = synth
        print("No pretraining checkpoint available; synthesized w2v_args from checkpoint weights:")
        print(f"  {synth.model}")

    # Initialize task and build model
    # Ensure task.data is valid to prevent Fairseq from reading stale absolute paths
    try:
        cfg.task.data = data_dir
    except Exception:
        pass
    # Match training-time waveform normalization (layer_norm over the clip)
    try:
        NORMALIZE = bool(getattr(cfg.task, "normalize", False))
    except Exception:
        NORMALIZE = False
    try:
        task = tasks.setup_task(cfg.task, from_checkpoint=True)
    except Exception as e:
        # The checkpoint may come from a custom task that is not registered in
        # this fairseq (e.g. trained on a cluster with newer task code). We only
        # need embeddings, so fall back to a plain audio_finetuning task and
        # rebuild the inner wav2vec2 from the config embedded in the checkpoint.
        print(f"setup_task('{getattr(cfg.task, '_name', '?')}') failed: {e}\n"
              f"→ falling back to audio_finetuning task")
        from fairseq.tasks.audio_finetuning import AudioFinetuningTask, AudioFinetuningConfig
        ft_cfg = AudioFinetuningConfig(data=data_dir, labels="ltr")
        task = AudioFinetuningTask.setup_task(ft_cfg)
    argspec = inspect.getfullargspec(task.build_model)
    if "from_checkpoint" in argspec.args:
        model = task.build_model(cfg.model, from_checkpoint=True)
    else:
        model = task.build_model(cfg.model)

    # Remove any mismatched keys (e.g., CTC head proj) before loading weights
    model_state = model.state_dict()
    ckpt_state = state["model"]
    pruned_state = {}
    for k, v in ckpt_state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            pruned_state[k] = v
        # else drop key silently (e.g., classifier head shape mismatch)

    # Load filtered weights non-strictly
    model.load_state_dict(pruned_state, strict=False)
    missing_w2v = [k for k in model_state if k not in pruned_state and "w2v_model" in k]
    print(f"Loaded {len(pruned_state)}/{len(model_state)} tensors from checkpoint")
    if missing_w2v:
        print(f"WARNING: {len(missing_w2v)} backbone tensors NOT found in checkpoint "
              f"(random init!), e.g. {missing_w2v[:5]}")

    # Move to device and eval
    model = model.to(DEVICE).eval()

    # Optionally restore num_updates for modules that care
    try:
        if (
            "optimizer_history" in state
            and state["optimizer_history"]
            and "num_updates" in state["optimizer_history"][-1]
        ):
            model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
    except Exception:
        pass

    MODEL, TASK = model, task


def _resample_float32(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    # Try scipy polyphase first for quality
    try:
        from scipy.signal import resample_poly
        # Use gcd for up/down to keep integers
        import math
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(wav, up, down).astype(np.float32, copy=False)
    except Exception:
        pass
    # Fallback to linear interpolation
    x = np.arange(wav.shape[0], dtype=np.float64)
    new_len = int(round(wav.shape[0] * (target_sr / float(orig_sr))))
    if new_len <= 1:
        return np.zeros((max(1, new_len),), dtype=np.float32)
    xp = np.linspace(0, wav.shape[0] - 1, num=wav.shape[0], dtype=np.float64)
    x_new = np.linspace(0, wav.shape[0] - 1, num=new_len, dtype=np.float64)
    y = np.interp(x_new, xp, wav.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)


def _trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    """Drop leading/trailing silence.

    Embeddings are mean-pooled over the whole clip, so silence dilutes them:
    a word recorded in a fixed 3-second window is mostly silence, and scores
    far below the same word cropped tightly. Measured on reference recordings,
    a 2 s leading gap cost 0.28 of cosine similarity. Keeping everything within
    TRIM_DB of the loudest 10 ms frame removes the penalty entirely - a word
    captured in a 3 s window then scores within 0.01 of the same word cropped
    tightly.
    """
    if not TRIM_SILENCE or wav.size < sr // 20:  # <50 ms: nothing to trim
        return wav
    win = max(1, int(sr * 0.01))
    n = wav.size // win
    if n < 2:
        return wav
    rms = np.sqrt((wav[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-12
    peak = rms.max()
    if peak <= 1e-9:  # all silence — leave it alone rather than return nothing
        return wav
    voiced = np.flatnonzero(20.0 * np.log10(rms / peak) > TRIM_DB)
    if voiced.size == 0:
        return wav
    pad = int(sr * TRIM_PAD_MS / 1000.0)
    a = max(0, voiced[0] * win - pad)
    b = min(wav.size, (voiced[-1] + 1) * win + pad)
    return wav[a:b] if b > a else wav


def _prep_wav(wav: np.ndarray, sr: int) -> Tuple[torch.Tensor, int]:
    # Ensure mono
    if hasattr(wav, 'ndim') and wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = _resample_float32(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Trim before normalizing: silence would otherwise skew the layer_norm too.
    wav = _trim_silence(wav, sr)
    x = torch.from_numpy(np.ascontiguousarray(wav)).float()
    # Ensure 1-D shape and non-empty content (pad very short audio)
    if x.dim() != 1:
        x = x.view(-1)
    if x.numel() < 1600:  # <0.1s at 16k → pad to 0.2s
        pad = 3200 - int(x.numel()) if x.numel() < 3200 else 0
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
    if NORMALIZE:  # same layer_norm applied by fairseq audio datasets at train time
        x = torch.nn.functional.layer_norm(x, x.shape)
    return x, sr


def _read_wav_from_upload(file: UploadFile) -> Tuple[torch.Tensor, int]:
    data = file.file.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    return _prep_wav(wav, sr)


def _read_wav_from_path(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32")
    return _prep_wav(wav, sr)


def _load_word_templates() -> dict:
    """Load the reference word library {word: [abs wav paths]} from the CSV
    manifest (wav_file,word,label; label==1 rows only). Cached after first load."""
    global WORD_TEMPLATES
    if WORD_TEMPLATES is not None:
        return WORD_TEMPLATES
    manifest = REFERENCE_MANIFEST
    if not manifest:
        cand = _HERE / "template_manifest_all_sources_with_xlsx.csv"
        manifest = str(cand) if cand.exists() else ""
    mapping: dict = {}
    prefixes = [] if REFERENCE_SOURCES == "*" else [
        s.strip().strip("/") for s in REFERENCE_SOURCES.split(",") if s.strip()
    ]
    skipped = 0
    if manifest and Path(manifest).exists():
        root = Path(REFERENCE_AUDIO_ROOT) if REFERENCE_AUDIO_ROOT else Path(manifest).resolve().parent
        import csv
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("label", "")).strip() != "1":
                    continue
                word = (row.get("word") or "").strip()
                wav = (row.get("wav_file") or row.get("wav") or "").strip()
                if not word or not wav:
                    continue
                if prefixes and not any(
                    wav == p or wav.startswith(p + "/") for p in prefixes
                ):
                    skipped += 1
                    continue
                p = root / wav
                if not p.exists():
                    p_wav = root / (wav + ".wav")
                    if not p_wav.exists():
                        continue
                    p = p_wav
                mapping.setdefault(word, []).append(str(p))
        src = "every source" if REFERENCE_SOURCES == "*" else REFERENCE_SOURCES
        note = f", {skipped} recordings from other sources skipped" if skipped else ""
        print(f"Reference library: {len(mapping)} words from {manifest} [{src}{note}]")
    else:
        print("Reference library: no manifest found (set REFERENCE_MANIFEST); dropdown disabled")
    WORD_TEMPLATES = mapping
    return mapping


def _get_inner_w2v_model():
    """Locate the wav2vec2 backbone so we can read intermediate transformer layers."""
    m = MODEL
    if hasattr(m, "w2v_encoder") and hasattr(m.w2v_encoder, "w2v_model"):
        return m.w2v_encoder.w2v_model
    if hasattr(m, "encoder") and hasattr(getattr(m, "encoder"), "layers") and hasattr(m, "extract_features"):
        return m
    return None


def _num_transformer_layers() -> Optional[int]:
    w2v = _get_inner_w2v_model()
    try:
        return len(w2v.encoder.layers)
    except Exception:
        return None


def _parse_layers_spec(spec) -> Optional[List[int]]:
    """Parse a layer spec like "20" or "19,20,21" into a list of 1-based ints.
    "" / None / "-1" → None (final encoder output)."""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s or s == "-1":
        return None
    layers = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            layers.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid layer spec {spec!r}: {part!r} is not an integer")
    return layers or None


@torch.no_grad()
def _extract_embedding(audio_16k: torch.Tensor, layers: Optional[List[int]] = None) -> np.ndarray:
    """Return an L2-normalized, mean-pooled embedding (1D input tensor).

    layers: list of 1-based transformer layer indices. Multiple layers are
    fused as in eval/predict_pron_correctness_multitemplate.py: per-layer
    mean-pooled embeddings are L2-normalized, averaged, then re-normalized.
    None → final encoder output.
    """
    assert audio_16k.dim() == 1
    audio = audio_16k.unsqueeze(0).to(DEVICE)
    m = MODEL
    n_layers = _num_transformer_layers()
    specs = layers if layers else [-1]
    for L in specs:
        if L != -1 and n_layers is not None and not (1 <= L <= n_layers):
            raise ValueError(f"layer must be in [1, {n_layers}], got {L}")

    feats_per_layer: List[torch.Tensor] = []
    if hasattr(m, "w2v_encoder"):
        pm = torch.zeros((1, audio.size(1)), device=DEVICE, dtype=torch.bool)
        enc = m.w2v_encoder(source=audio, padding_mask=pm, mask=False, features_only=True)
        layer_results = enc.get("layer_results", None) if isinstance(enc, dict) else None
        for L in specs:
            if L == -1:
                feat = enc["encoder_out"]  # (T, B, C)
            else:
                if not layer_results:
                    raise RuntimeError("Model did not return layer_results; use the final layer")
                feat = layer_results[L - 1][0]  # (T, B, C), output of transformer block L
            if feat.dim() == 3:
                feat = feat[:, 0, :]  # (T, C)
            feats_per_layer.append(feat)
    else:
        w2v = _get_inner_w2v_model()
        if w2v is None:
            raise RuntimeError("Cannot locate a wav2vec2 encoder on the loaded model")
        res = w2v.extract_features(audio, padding_mask=None, mask=False)
        layer_results = res.get("layer_results", None) if isinstance(res, dict) else None
        for L in specs:
            if L == -1:
                feat = res["x"]  # (B, T, C)
                feat = feat[0]
            else:
                if not layer_results:
                    raise RuntimeError("Model did not return layer_results; use the final layer")
                feat = layer_results[L - 1][0][:, 0, :]
            feats_per_layer.append(feat)

    embs: List[torch.Tensor] = []
    for feat in feats_per_layer:
        if feat.dim() != 2 or feat.size(0) == 0:
            raise RuntimeError(f"Unexpected features shape: {tuple(feat.shape)}")
        v = feat.float().mean(dim=0)
        v = v / (v.norm() + 1e-12)
        embs.append(v)
    fused = torch.stack(embs, dim=0).mean(dim=0)
    fused = fused / (fused.norm() + 1e-12)
    emb = fused.detach().cpu().numpy()
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)


def _prototype(embs: List[np.ndarray]) -> np.ndarray:
    """Average multiple (already normalized) embeddings into one prototype."""
    proto = np.mean(np.stack(embs, axis=0), axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-12)
    return proto


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine expects 1D vectors")
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    try:
        _load_model()
        return Health(status="ok")
    except Exception as e:
        print(f"Error loading model: {e}")
        return Health(status="error")


@app.get("/api/model_info")
def model_info():
    try:
        _load_model()
        return {
            "num_layers": _num_transformer_layers(),
            # None → final encoder output ("" or "-1" both mean final)
            "default_layer": None if EMBEDDING_LAYERS in ("", "-1") else EMBEDDING_LAYERS,
            "device": str(DEVICE),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/words")
def list_words():
    """List reference-library words available for the dropdown."""
    try:
        words = _load_word_templates()
        return {"words": [{"word": w, "n_templates": len(v)} for w, v in sorted(words.items())]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/word_audio")
def word_audio(word: str, i: int = 0):
    """Serve the i-th template recording of a library word (for preview)."""
    from fastapi.responses import FileResponse
    words = _load_word_templates()
    paths = words.get((word or "").strip())
    if not paths:
        return JSONResponse(status_code=404, content={"error": f"Unknown word: {word!r}"})
    i = max(0, min(int(i), len(paths) - 1))
    return FileResponse(paths[i], media_type="audio/wav", filename=Path(paths[i]).name)


def _word_template_embeddings(word: str, layers, layers_key):
    """Return (cached) per-template embeddings for a library word, building the
    prototype set once. Raises ValueError if the word is unknown / unreadable."""
    templates = _load_word_templates().get(word)
    if not templates:
        raise ValueError(f"Unknown reference word: {word!r}")
    cache_key = (word, layers_key)
    if cache_key not in PROTO_CACHE:
        embs = []
        skipped = 0
        for p in templates:
            try:
                wav, _ = _read_wav_from_path(p)
            except Exception:
                try:
                    Path(p).read_bytes()  # force materialization (e.g. iCloud dataless file)
                    wav, _ = _read_wav_from_path(p)
                except Exception as e2:
                    skipped += 1
                    print(f"[warn] word '{word}': skipping unreadable template {p}: {e2}")
                    continue
            embs.append(_extract_embedding(wav, layers=layers))
        if not embs:
            raise ValueError(f"No readable template recordings for word {word!r}")
        if skipped:
            print(f"[warn] word '{word}': prototype built from {len(embs)} templates ({skipped} skipped)")
        PROTO_CACHE[cache_key] = embs
    return PROTO_CACHE[cache_key]


@app.post("/api/batch_score")
def batch_score(
    files: List[UploadFile] = File(...),
    words: List[str] = Form(...),
    layer: Optional[str] = Form(None),
):
    """Score a batch of candidate recordings against the reference word library.
    `files[i]` is scored against the library prototype for `words[i]`.
    Returns one result row per file (unknown words are flagged, not fatal)."""
    try:
        _load_model()
        if len(files) != len(words):
            raise ValueError(f"files ({len(files)}) and words ({len(words)}) length mismatch")
        layers = _parse_layers_spec(layer) if layer is not None else _parse_layers_spec(EMBEDDING_LAYERS)
        layers_key = tuple(layers) if layers else ("final",)
        layer_str = ",".join(str(x) for x in layers) if layers else "final"

        results = []
        for f, w in zip(files, words):
            w = (w or "").strip()
            row = {"filename": f.filename, "word": w, "score": None, "in_library": False, "n_templates": 0, "error": None}
            try:
                proto_embs = _word_template_embeddings(w, layers, layers_key)
                wav, _ = _read_wav_from_upload(f)
                cand = _extract_embedding(wav, layers=layers)
                cos = _cosine(_prototype(proto_embs), cand)
                if not np.isfinite(cos):
                    cos = 0.0
                row.update(score=float(cos), in_library=True, n_templates=len(proto_embs))
            except Exception as e:
                row["error"] = str(e)
            results.append(row)

        scored = [r["score"] for r in results if r["score"] is not None]
        return {
            "results": results,
            "layers": layer_str,
            "n_files": len(files),
            "n_scored": len(scored),
            "n_skipped": len(files) - len(scored),
            "mean_score": float(np.mean(scored)) if scored else None,
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=400, content={"error": str(e), "trace": traceback.format_exc()})


@app.post("/api/similarity")
def similarity(
    candidate: List[UploadFile] = File(...),
    reference: Optional[List[UploadFile]] = File(None),
    reference_word: Optional[str] = Form(None),
    layer: Optional[str] = Form(None),
):
    """Multi-template similarity. Reference templates come from either or both:
    uploaded 'reference' files, and/or 'reference_word' — a word from the
    library whose label==1 recordings all become templates. Every template is
    embedded separately and averaged into an L2-normalized prototype; the
    score is the cosine between the reference and candidate prototypes.

    layer: 1-based layer spec, e.g. "20" or "19,20,21" (multi-layer fusion).
    """
    try:
        _load_model()
        # Per-request layer spec overrides EMBEDDING_LAYER env; None → final output
        layers = _parse_layers_spec(layer) if layer is not None else _parse_layers_spec(EMBEDDING_LAYERS)
        layers_key = tuple(layers) if layers else ("final",)

        ref_embs = []
        n_library = 0
        word = (reference_word or "").strip()
        if word:
            proto_embs = _word_template_embeddings(word, layers, layers_key)
            ref_embs.extend(proto_embs)
            n_library = len(proto_embs)
        for f in (reference or []):
            wav, _ = _read_wav_from_upload(f)
            ref_embs.append(_extract_embedding(wav, layers=layers))
        if not ref_embs:
            raise ValueError("Provide at least one 'reference' file or a 'reference_word'")

        cand_embs = []
        for f in candidate:
            wav, _ = _read_wav_from_upload(f)
            cand_embs.append(_extract_embedding(wav, layers=layers))
        if not cand_embs:
            raise ValueError("Need at least one candidate file")

        ref_emb = _prototype(ref_embs)
        cand_emb = _prototype(cand_embs)

        cos = _cosine(ref_emb, cand_emb)
        if not np.isfinite(cos):
            cos = 0.0
        # Score IS the raw cosine similarity, in [-1, 1]
        score = cos

        layer_str = ",".join(str(x) for x in layers) if layers else "final"
        ref_desc = f"ref_templates={len(ref_embs)}"
        if word:
            ref_desc += f" (word='{word}': {n_library} library + {len(ref_embs) - n_library} uploaded)"
        return {
            "score": float(score),
            "cosine": float(cos),
            "n_reference_templates": len(ref_embs),
            "n_candidate_templates": len(cand_embs),
            "reference_word": word or None,
            "explain": f"emb_dim={ref_emb.shape[0]} layers={layer_str} "
                       f"{ref_desc} cand_templates={len(cand_embs)} device={DEVICE}",
        }
    except Exception as e:
        # Also log full trace on server for quick diagnosis
        print(traceback.format_exc())
        return JSONResponse(status_code=400, content={
            "error": str(e),
            "trace": traceback.format_exc(),
        })


# Serve the built frontend (GUI/dist) if present, so a single process hosts
# both the web app (/) and the API (/api, /healthz). Mounted last so the
# explicit API routes above take precedence. Skipped if no build is present
# (e.g. running the API alone behind a Vite dev server).
_STATIC_DIR = os.environ.get("STATIC_DIR", str(_HERE / "GUI" / "dist"))
if os.path.isdir(_STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    print(f"Serving frontend from {_STATIC_DIR}")
else:
    print(f"No frontend build at {_STATIC_DIR}; serving API only")


if __name__ == "__main__":
    import uvicorn
    # Read optional host/port from env for convenience
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api_similarity_server:app", host=host, port=port, reload=False)
