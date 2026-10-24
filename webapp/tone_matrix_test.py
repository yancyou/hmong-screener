#!/usr/bin/env python3
"""Tone-separability probe on the 'dhia' minimal-pair family.
For each word, build a prototype from its _tone_female + _tone_male recordings
and use the base recording as a held-out candidate. Score every candidate
against every prototype -> matrix. Tone-separable => diagonal wins each row.

Env: CHECKPOINT_PATH, EMBEDDING_LAYER (-1=final, or 20, or 19,20,21)
"""
import os
import numpy as np

import api_similarity_server as s

WORDS = [("dhiab", "036"), ("dhia", "037"), ("dhias", "038"), ("dhiaj", "039")]
AD = "all_data/Hmong_Word_List_1a_Recording_word{w}{s}.wav"


def emb(path, layers):
    wav, _ = s._read_wav_from_path(path)
    return s._extract_embedding(wav, layers=layers)


def main():
    s._load_model()
    spec = os.environ.get("EMBEDDING_LAYER", "-1").strip()
    layers = s._parse_layers_spec(spec)
    lbl = "final" if not layers else ",".join(map(str, layers))

    protos, cands = [], []
    for name, w in WORDS:
        p = s._prototype([emb(AD.format(w=w, s="_tone_female"), layers),
                          emb(AD.format(w=w, s="_tone_male"), layers)])
        protos.append(p)
        cands.append(emb(AD.format(w=w, s=""), layers))

    print(f"\n=== checkpoint={os.path.basename(os.environ['CHECKPOINT_PATH'])}  layer={lbl} ===")
    print("candidate \\ proto  " + "  ".join(f"{n:>7}" for n, _ in WORDS))
    correct = 0
    for i, (name, _) in enumerate(WORDS):
        row = [float(np.dot(cands[i], protos[j]) /
                     ((np.linalg.norm(cands[i]) + 1e-12) * (np.linalg.norm(protos[j]) + 1e-12)))
               for j in range(len(WORDS))]
        best = int(np.argmax(row))
        ok = best == i
        correct += ok
        marks = "  ".join((f"[{v:+.3f}]" if j == i else f" {v:+.3f} ") for j, v in enumerate(row))
        print(f"{name:>10}  cand   {marks}   -> best={WORDS[best][0]} {'OK' if ok else 'XX'}")
    print(f"diagonal-wins: {correct}/{len(WORDS)}  (higher = better tone separation)")


if __name__ == "__main__":
    main()
