#!/usr/bin/env python3
"""部署候选组合评估 (向量化)。每个组合 = 若干模型 CTC loss 的平均；打分同部署 (候选全集+按折无泄漏先验)。
   指标: orig32 同一批 clip 的整词/声调 AUC、检出率、top-1；声调另算 +F0 会话内秩融合(部署口径)。"""
import json,csv,numpy as np
from scipy.stats import rankdata
from score_losses import score_fast, loss_path
import os
E=os.environ.get("HMONG_EVALS","/staging/x/xouyang7/evals33"); os.makedirs("build",exist_ok=True)
TAG=os.environ.get("HMONG_OUT_TAG","")   # suffix for output files, e.g. "_fixed"
F={"v1":"v1","v2":"v2","s1b":"v3bbest","s1l":"v3blast","s1a":"v3bavg5","s2b":"v3bs2best","s2l":"v3bs2last","s2a":"v3bs2avg5"}
COMBOS={"v1":["v1"],"s1b":["s1b"],"ens_s1b+s2b":["s1b","s2b"],
 "v1+s1b":["v1","s1b"],"v1+s2b":["v1","s2b"],"v1+s1b+s2b":["v1","s1b","s2b"],
 "v1+s1l+s2l":["v1","s1l","s2l"],"v1+s1a+s2a":["v1","s1a","s2a"],"v1+全部6个v3":["v1","s1b","s1l","s1a","s2b","s2l","s2a"],
 "v1+v2":["v1","v2"],"v1+v2+s1b+s2b":["v1","v2","s1b","s2b"]}
idx={f"full32_16k/{r['wav_file']}":r for r in csv.DictReader(open('data/full33_index.csv'))}
orig={f"full32_16k/{r['wav_file']}" for r in csv.DictReader(open('data/full32_index.csv'))}
S={c:{"Pw":[],"Pt":[],"hit":[]} for c in COMBOS}; wav=[]; lab=[]; spk=[]
for k in range(4):
    pr=f"data/priors33/priors_fold{k}.json"; L={}; d0=None
    for nm,f in F.items():
        d=np.load(loss_path(E,f,k)); d0=d if d0 is None else d0
        assert (d["wav"]==d0["wav"]).all(); L[nm]=d["loss"]
    for c,parts in COMBOS.items():
        Pw,Pt,hit=score_fast(d0,sum(L[p] for p in parts)/len(parts),pr)
        S[c]["Pw"].append(Pw); S[c]["Pt"].append(Pt); S[c]["hit"].append(hit)
    wav+=list(d0["wav"]); lab+=list(d0["label"]); spk+=list(d0["speaker"]); del L
    print(f"fold{k} ok",flush=True)
wav=np.array(wav); lab=np.array(lab); spk=np.array(spk)
for c in S: S[c]={k:np.concatenate(v) for k,v in S[c].items()}
U=np.array([idx[w]["src"]=="union-err" for w in wav]); O=np.array([w in orig for w in wav])
# F0 后验 (32 人, 按折留出训练) 对齐
m=np.load('data/derived/f0_full32_meta.npz',allow_pickle=True); pf0=dict(zip(m['src'],np.load('data/derived/P_f0_full32.npy')))
F0=np.array([pf0.get(w,np.nan) for w in wav])
def auc(s,y):
    p=int(y.sum()); n=int((~y).sum())
    if p<5 or n<5: return np.nan
    r=rankdata(-s); return (r[y].sum()-p*(p+1)/2)/(p*n)
def fused_session(Pt,sel_mask):
    """会话内秩融合: 每个说话人内部把 CTC 声调分与 F0 分各自转成秩再按 0.6/0.4 加权 (与 deploy_screen 一致)"""
    per=[]; pooled_s=np.full(len(Pt),np.nan)
    for s in np.unique(spk[sel_mask]):
        mm=sel_mask&(spk==s)&~np.isnan(F0)
        if mm.sum()<20: continue
        f=0.6*rankdata(Pt[mm])/mm.sum()+0.4*rankdata(F0[mm])/mm.sum(); pooled_s[mm]=f
        y=lab[mm]==2; ok=lab[mm]==1; sel=y|ok
        if y.sum()>=5: per.append(auc(f[sel],y[sel]))
    return np.nanmean(per),np.nanmin(per),per
out={}
base=O
print(f"\n=== orig32 同一批 clip (整词错 {int(((lab==3)&O).sum())} / 声调错 {int(((lab==2)&O).sum())}) —— 按 [整词AUC+融合后声调] 排序 ===")
print(f"{'组合':<16}{'整词AUC':>8}{'(可靠)':>8}{'整词检出':>8}{'声调AUC':>8}{'融合声调(人均)':>13}{'(最差人)':>9}{'top1':>7}")
for c in COMBOS:
    Pw,Pt,hit=S[c]["Pw"],S[c]["Pt"],S[c]["hit"]
    ok=(lab==1)&O; w=(lab==3)&O; t=(lab==2)&O
    r={"word":auc(Pw[w|ok],w[w|ok]),"word_rel":auc(Pw[(w&~U)|ok],(w&~U)[(w&~U)|ok]),
       "word_rec10":float((Pw[w]<np.sort(Pw[ok])[int(.1*ok.sum())]).mean()),
       "tone":auc(Pt[t|ok],t[t|ok]),"top1":float(hit[ok].mean())}
    r["tone_f0_mean"],r["tone_f0_worst"],_=fused_session(Pt,O)
    out[c]=r
for c in sorted(out,key=lambda c:-(out[c]["word"]+out[c]["tone_f0_mean"])):
    r=out[c]; print(f"{c:<16}{r['word']:>8.4f}{r['word_rel']:>8.4f}{r['word_rec10']:>8.1%}{r['tone']:>8.4f}{r['tone_f0_mean']:>13.4f}{r['tone_f0_worst']:>9.3f}{r['top1']:>7.4f}")
json.dump(out,open(f'build/eval_combos{TAG}.json','w'),indent=1)
np.savez(f'build/eval_combos_scores{TAG}.npz',wav=wav,lab=lab,spk=spk,U=U,O=O,F0=F0,**{f"{c}__{k}":v for c in S for k,v in S[c].items()})
