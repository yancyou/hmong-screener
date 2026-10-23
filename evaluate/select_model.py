#!/usr/bin/env python3
"""部署选型: 所有候选在同一批留出录音上、部署同款候选全集 + 按折无泄漏先验打分。
   单模型: v1, v2, v3b{best,last,avg5}(种子1), v3bs2{best,last,avg5}(种子2)
   集成  : 同规则两种子 CTC loss 平均 (s1+s2)，以及 6 个 v3 全部平均
   一次只装一折的 loss 矩阵(登录节点内存有限)。输出 select_model.json"""
import json,csv,numpy as np
from scipy.stats import rankdata
from score_losses import score_arrays, base_of, loss_path
import os
E=os.environ.get("HMONG_EVALS","/staging/x/xouyang7/evals33"); os.makedirs("build",exist_ok=True)
TAG=os.environ.get("HMONG_OUT_TAG","")   # suffix for output files, e.g. "_fixed"
SINGLE={"v1":"v1","v2":"v2","s1_best":"v3bbest","s1_last":"v3blast","s1_avg5":"v3bavg5",
        "s2_best":"v3bs2best","s2_last":"v3bs2last","s2_avg5":"v3bs2avg5"}
ENS={"ens_best":["s1_best","s2_best"],"ens_last":["s1_last","s2_last"],"ens_avg5":["s1_avg5","s2_avg5"],
     "ens_all6":["s1_best","s1_last","s1_avg5","s2_best","s2_last","s2_avg5"],"ens_v1v2":["v1","v2"]}
idx={f"full32_16k/{r['wav_file']}":r for r in csv.DictReader(open('data/full33_index.csv'))}
orig={f"full32_16k/{r['wav_file']}" for r in csv.DictReader(open('data/full32_index.csv'))}
R={k:[] for k in list(SINGLE)+list(ENS)}
for k in range(4):
    pr=f"data/priors33/priors_fold{k}.json"; L={}; d0=None
    for name,f in SINGLE.items():
        d=np.load(loss_path(E,f,k))
        if d0 is None: d0=d
        assert (d["wav"]==d0["wav"]).all() and (d["cand"]==d0["cand"]).all(), (name,k)
        L[name]=d["loss"]
    for name in SINGLE: R[name]+=score_arrays(d0,L[name],pr)
    for name,parts in ENS.items(): R[name]+=score_arrays(d0,sum(L[p] for p in parts)/len(parts),pr)
    print(f"fold{k} done",flush=True); del L
wav=[r["wav"] for r in R["v1"]]
for n in R: assert [r["wav"] for r in R[n]]==wav
lab=np.array([r["label"] for r in R["v1"]]); spk=np.array([r["spk"] for r in R["v1"]])
U=np.array([idx[w]["src"]=="union-err" for w in wav]); isorig=np.array([w in orig for w in wav])
A={n:dict(Pw=np.array([r["P_word"] for r in R[n]]),Pt=np.array([r["P_tone"] for r in R[n]]),
          hit=np.array([base_of(r["top1"])==base_of(r["word"]) for r in R[n]])) for n in R}
def auc(s,y):
    p=int(y.sum()); n=int((~y).sum()); r=rankdata(-s); return (r[y].sum()-p*(p+1)/2)/(p*n)
def met(a,ii):
    l=lab[ii]; u=U[ii]; ok=l==1; m={}
    for nm,s,pl in (("word",a["Pw"][ii],3),("tone",a["Pt"][ii],2)):
        for tag,pos in (("all",l==pl),("rel",(l==pl)&~u)):
            sel=pos|ok; m[f"{nm}_{tag}"]=auc(s[sel],pos[sel])
        t=np.sort(s[ok])[int(0.10*ok.sum())]; m[f"{nm}_rec10"]=float((s[l==pl]<t).mean())
    m["top1"]=float(a["hit"][ii][ok].mean())
    m["mean_auc"]=(m["word_all"]+m["tone_all"])/2
    return m
out={}
for sub,mask in (("orig32",isorig),("all33",np.ones(len(wav),bool))):
    ii=np.where(mask)[0]; out[sub]={n:met(A[n],ii) for n in A}
    print(f"\n=== {sub}  (整词错 {int((lab[ii]==3).sum())} / 声调错 {int((lab[ii]==2).sum())} / 正确 {int((lab[ii]==1).sum())}) ===")
    print(f"{'候选':<10}{'整词AUC':>8}{'(可靠)':>8}{'整词检出':>8}{'声调AUC':>8}{'(可靠)':>8}{'声调检出':>8}{'top1':>7}{'平均AUC':>8}")
    for n in sorted(out[sub],key=lambda n:-out[sub][n]["mean_auc"]):
        m=out[sub][n]
        print(f"{n:<10}{m['word_all']:>8.4f}{m['word_rel']:>8.4f}{m['word_rec10']:>8.1%}{m['tone_all']:>8.4f}{m['tone_rel']:>8.4f}{m['tone_rec10']:>8.1%}{m['top1']:>7.4f}{m['mean_auc']:>8.4f}")
json.dump(out,open(f"build/select_model{TAG}.json","w"),indent=1)
np.savez(f"build/select_model_scores{TAG}.npz",wav=np.array(wav),lab=lab,spk=spk,U=U,isorig=isorig,
         **{f"{n}__Pw":A[n]["Pw"] for n in A},**{f"{n}__Pt":A[n]["Pt"] for n in A},**{f"{n}__hit":A[n]["hit"] for n in A})
print("\n→ select_model.json / select_model_scores.npz")
