#!/usr/bin/env python3
"""Compare a recomputed held-out loss file with the stored one, row by row (matched on wav names).
Usage: python3 tests/compare_losses.py <new.npz> <stored.npz>"""
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
assert list(a["cand"]) == list(b["cand"]), "candidate lists differ"
ib = {w: i for i, w in enumerate(b["wav"])}
rows = [(i, ib[w]) for i, w in enumerate(a["wav"]) if w in ib]
A = a["loss"][[i for i, _ in rows]]; B = b["loss"][[j for _, j in rows]]
fin = np.isfinite(A) & np.isfinite(B)
rel = np.abs(A[fin] - B[fin]) / np.maximum(1.0, np.abs(B[fin]))
print(f"{len(rows)} clips x {A.shape[1]} candidates | max relative difference {rel.max():.2e} | "
      f"argmin candidate agrees on {np.mean(A.argmin(1) == B.argmin(1))*100:.2f}% of clips")
sys.exit(0 if rel.max() < 1e-2 else 1)
