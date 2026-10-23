#!/usr/bin/env python3
"""Train the deployed pitch (F0) tone classifier, deploy/assets/f0_tone_clf.npz.
Recovered from the session log of 2026-08-25 (the run that produced the deployed file); logic unchanged.
A 7-class softmax regression on the 25 contour features from extract_f0_features.py, fitted on every correct clip
(label 1) of the 32 speakers with plain gradient descent: 800 epochs, learning rate 0.5, L2 3e-4, standardized inputs.
Input : build/f0/f0_shard*.npz + .csv (from f0_extract.sub), data/full32_merged.csv (pre-merge `label` column)
Output: build/f0_tone_clf.npz (W, b, mu, sd)
Usage : python3 evaluate/f0/train_f0_classifier.py"""
import numpy as np, csv, glob, os
lab = {}
for r in csv.DictReader(open('data/full32_merged.csv')):
    if r['label'] in ('1', '2', '3'):
        lab[r['wav_file']] = int(r['label']); lab['full32_16k/' + r['wav_file']] = int(r['label'])
X = []; yt = []; sp = []; src = []
for p in sorted(glob.glob('build/f0/f0_shard*.npz')):
    d = np.load(p, allow_pickle=True)
    rows = list(csv.DictReader(open(p.replace('.npz', '.csv'))))
    if len(rows) != len(d['X']): print(f"  skip {p}: row count differs"); continue
    X.append(d['X']); yt.append(d['y_tone']); sp.append(d['speaker']); src += [r['wav_file'] for r in rows]
X = np.concatenate(X); yt = np.concatenate(yt); sp = np.concatenate(sp); src = np.array(src)
keep = np.array([w in lab for w in src]); X, yt, sp, src = X[keep], yt[keep], sp[keep], src[keep]
y3 = np.array([lab[w] for w in src])
print(f"{len(X)} clips / {len(set(sp))} speakers with features and labels")
mu, sd = X.mean(0), X.std(0) + 1e-9; Xn = (X - mu) / sd
tr = y3 == 1
def sm_train(Xt, y, K, ep=800, lr=0.5, l2=3e-4):
    W = np.zeros((Xt.shape[1], K)); b = np.zeros(K); Y = np.eye(K)[y]
    for _ in range(ep):
        z = Xt @ W + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(y); W -= lr * (Xt.T @ g + l2 * W); b -= lr * g.sum(0)
    return W, b
W, b = sm_train(Xn[tr], yt[tr] - 1, 7)
acc = ((Xn[tr] @ W + b).argmax(1) == (yt[tr] - 1)).mean()
print(f"training accuracy on {tr.sum()} correct clips: {acc:.3f}")
os.makedirs('build', exist_ok=True)
np.savez('build/f0_tone_clf.npz', W=W, b=b, mu=mu, sd=sd,
         note="F0 tone classifier; input=25d contour feats; output=7 tone posteriors (b,none,s,j,v,g,m order per TONE map)")
print("saved build/f0_tone_clf.npz")
