#!/usr/bin/env python3
"""
Extract speaker-normalized F0-contour features for every labeled sample.

Tone = pitch contour over time. We:
  1. pYIN F0 per frame (voiced only)
  2. speaker-normalize log-F0 (z-score within speaker's voiced range) -> speaker-invariant
  3. summarize the normalized contour into a fixed feature vector + phonation cues

Output: f0_features.npz with X [N,D], y_tone [N], y_wrong [N], speaker [N], word [N].

CPU only, multiprocessing. ~20 min on 8 cores for 17k clips.

Usage:
  python extract_f0_features.py --labels-csv template_manifest_all_xlsx_speakers.csv \
     --data-dir . --out f0_features.npz --workers 8
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import librosa

TONE={"b":1,"":2,"s":3,"j":4,"v":5,"g":6,"m":7}
VALID=set("bsjvgm")
NPTS=16  # resample contour to this many points

def tone_of(w):
    w=(w or "").strip().lower()
    return TONE.get(w[-1],2) if w and w[-1] in VALID else 2

def load_rows(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            wav=(r.get("wav_file") or "").strip(); word=(r.get("word") or "").strip()
            lbl=(r.get("label") or "").strip(); spk=(r.get("speaker") or "").strip().lower()
            if not wav or not word or not lbl: continue
            try: li=int(lbl)
            except: continue
            if li==1: wrong=0
            elif li in (2,3,4): wrong=1
            else: continue
            rows.append({"wav":wav,"word":word,"spk":spk,"wrong":wrong,"tone":tone_of(word)})
    return rows

def extract_raw_f0(args):
    """Return (idx, logf0_array_voiced_only, voiced_ratio, mean_vprob) or (idx,None,..)."""
    idx, wav_full = args
    try:
        y,sr=librosa.load(wav_full,sr=16000)
        if y.size<800: return (idx,None,0.0,0.0)
        f0,vflag,vprob=librosa.pyin(y,fmin=70,fmax=400,sr=sr,frame_length=512)
        voiced=~np.isnan(f0)
        vr=float(voiced.mean()) if len(f0) else 0.0
        mvp=float(np.nanmean(vprob)) if vprob is not None and len(vprob) else 0.0
        f0v=f0[voiced]
        if len(f0v)<3: return (idx,None,vr,mvp)
        return (idx,np.log(f0v).astype(np.float32),vr,mvp)
    except Exception:
        return (idx,None,0.0,0.0)

def contour_features(logf0_norm):
    """Given speaker-normalized log-F0 (voiced), return fixed feature vector."""
    n=len(logf0_norm)
    x=np.linspace(0,1,n)
    # resample to NPTS
    xs=np.linspace(0,1,NPTS)
    resamp=np.interp(xs,x,logf0_norm)
    # polynomial fit (slope, curvature)
    c=np.polyfit(x,logf0_norm,2)  # [a2,a1,a0]
    slope=float(c[1]); curv=float(c[0])
    feats=np.concatenate([
        resamp,                                  # 16 contour points
        [float(logf0_norm.mean()), float(logf0_norm.std()),
         float(logf0_norm[0]), float(logf0_norm[-1]),
         float(logf0_norm.max()-logf0_norm.min()),
         slope, curv],
    ]).astype(np.float32)
    return feats  # 16+7 = 23

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels-csv",required=True)
    ap.add_argument("--data-dir",default=".")
    ap.add_argument("--out",required=True)
    ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args()

    rows=load_rows(a.labels_csv)
    print(f"rows: {len(rows)}")

    # resolve paths
    tasks=[]
    for i,r in enumerate(rows):
        wp=os.path.join(a.data_dir,r["wav"])
        if not os.path.exists(wp) and not r["wav"].lower().endswith(".wav") and os.path.exists(wp+".wav"):
            wp=wp+".wav"
        tasks.append((i,wp))

    print(f"extracting F0 with {a.workers} workers ...")
    raw=[None]*len(rows); vr=[0.0]*len(rows); mvp=[0.0]*len(rows)
    with Pool(a.workers) as p:
        for k,(idx,lf,v,m) in enumerate(p.imap_unordered(extract_raw_f0,tasks,chunksize=32),1):
            raw[idx]=lf; vr[idx]=v; mvp[idx]=m
            if k%2000==0: print(f"  {k}/{len(rows)}")

    # Speaker normalization stats: mean/std of log-F0 over ALL that speaker's voiced frames
    spk_vals=defaultdict(list)
    for i,r in enumerate(rows):
        if raw[i] is not None:
            spk_vals[r["spk"]].append(raw[i])
    spk_stat={}
    for s,lst in spk_vals.items():
        allv=np.concatenate(lst)
        spk_stat[s]=(float(allv.mean()),float(allv.std()+1e-6))
    print(f"speaker norm stats: {len(spk_stat)} speakers")

    # Build feature matrix
    X=[]; y_tone=[]; y_wrong=[]; spk=[]; word=[]; n_unvoiced=0
    for i,r in enumerate(rows):
        if raw[i] is None:
            n_unvoiced+=1
            # unvoiced/failed: zeros + flag via voiced_ratio
            feats=np.zeros(23+2,dtype=np.float32)
            feats[-2]=vr[i]; feats[-1]=mvp[i]
        else:
            mu,sd=spk_stat[r["spk"]]
            norm=(raw[i]-mu)/sd
            cf=contour_features(norm)
            feats=np.concatenate([cf,[vr[i],mvp[i]]]).astype(np.float32)  # +voiced_ratio,+mean_vprob = 25
        X.append(feats); y_tone.append(r["tone"]); y_wrong.append(r["wrong"])
        spk.append(r["spk"]); word.append(r["word"])

    X=np.stack(X); y_tone=np.array(y_tone); y_wrong=np.array(y_wrong)
    print(f"X: {X.shape}, unvoiced/failed: {n_unvoiced}")
    print(f"wrong: {int(y_wrong.sum())}, correct: {int((1-y_wrong).sum())}")
    np.savez(a.out, X=X, y_tone=y_tone, y_wrong=y_wrong,
             speaker=np.array(spk), word=np.array(word),
             feat_names=np.array([f"c{i}" for i in range(NPTS)]+
                ["mean","std","start","end","range","slope","curv","voiced_ratio","mean_vprob"]))
    print(f"saved {a.out}")

if __name__=="__main__":
    main()
