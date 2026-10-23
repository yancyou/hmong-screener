#!/usr/bin/env python3
"""v1 / v2 / v3 同一批留出 clip 上的对照 (全部用无泄漏先验) + v1 泄漏对照 + 说话人级 bootstrap。
   Usage (from the repo root): python3 evaluate/compare_models.py [--evals /staging/x/xouyang7/evals33] [--out build/compare_models.json]"""
import argparse,csv,json,os,collections
import numpy as np
from scipy.stats import rankdata
from score_losses import score_npz, base_of, loss_path
ap=argparse.ArgumentParser(); ap.add_argument("--evals",default="/staging/x/xouyang7/evals33")
ap.add_argument("--out",default="build/compare_models.json"); ap.add_argument("--boot",type=int,default=1000)
a=ap.parse_args()
idx={f"full32_16k/{r['wav_file']}":r for r in csv.DictReader(open('data/full33_index.csv'))}
orig32={f"full32_16k/{r['wav_file']}" for r in csv.DictReader(open('data/full32_index.csv'))}
def load(model,prior):
    rows=[]
    for k in range(4):
        p=loss_path(a.evals,model.replace('_',''),k)   # honours HMONG_FOLD_FILES
        if not os.path.exists(p): return None
        pr=f"data/priors33/priors_fold{k}.json" if prior=="clean" else "data/confusion_priors.json"
        for s in score_npz(p,pr): s["fold"]=k; rows.append(s)
    return rows
def auc(score,y):                                   # 分数越低越可能是错误
    p=int(y.sum()); n=int((~y).sum())
    if p<3 or n<3: return np.nan
    r=rankdata(-score); return (r[y].sum()-p*(p+1)/2)/(p*n)
def rec_at_fpr(score,pos,neg,cap=0.10):
    t=np.sort(score[neg])[int(cap*neg.sum())]; return float((score[pos]<t).mean())
def arrays(R):
    return dict(lab=np.array([r["label"] for r in R]),Pw=np.array([r["P_word"] for r in R]),
                Pt=np.array([r["P_tone"] for r in R]),src=np.array([idx[r["wav"]]["src"] for r in R]),
                hit_full=np.array([r["top1"]==r["word"] for r in R]),
                hit_base=np.array([base_of(r["top1"])==base_of(r["word"]) for r in R]))
def metrics(A,ii):                                  # ii: 索引数组(可重复, 供 bootstrap)
    lab=A["lab"][ii]; src=A["src"][ii]; ok=lab==1; m={}
    for nm,s,pl in (("word",A["Pw"][ii],3),("tone",A["Pt"][ii],2)):
        pos=lab==pl
        for tag,pp in (("all",pos),("clean",pos&(src!="union-err"))):
            sel=pp|ok; m[f"{nm}_AUC_{tag}"]=auc(s[sel],pp[sel]); m[f"{nm}_rec10_{tag}"]=rec_at_fpr(s,pp,ok)
            m[f"n_{nm}err_{tag}"]=int(pp.sum())
    m["top1_full"]=float(A["hit_full"][ii][ok].mean()); m["top1_base"]=float(A["hit_base"][ii][ok].mean())
    m["n_ok"]=int(ok.sum())
    return m
res={}; data={}
for model,prior in (("v1","clean"),("v2","clean"),("v3b_best","clean"),("v3b_last","clean"),("v3b_avg5","clean"),("v1","leaky")):
    R=load(model,prior)
    if R is None: print(f"[skip] {model}: loss 不全"); continue
    data[(model,prior)]=R
wav0=None
for key,R in data.items():                          # 所有模型必须是同一批 clip、同一顺序
    w=[r["wav"] for r in R]
    if wav0 is None: wav0=w
    assert w==wav0, f"{key} clip 列表不一致"
if not data: raise SystemExit("没有可比较的结果")
isorig=np.array([w in orig32 for w in wav0]); spk=np.array([r["spk"] for r in next(iter(data.values()))])
subsets={"orig32 (与旧结果同一批clip)":isorig,"new (2026-09 新增)":~isorig,"all33":np.ones(len(wav0),bool)}
ARR={key:arrays(R) for key,R in data.items()}
for key in data:
    res[f"{key[0]}_{key[1]}"]={sn:metrics(ARR[key],np.where(sm)[0]) for sn,sm in subsets.items()}
keys=["word_AUC_all","word_rec10_all","word_AUC_clean","word_rec10_clean","tone_AUC_all","tone_rec10_all","tone_AUC_clean","tone_rec10_clean","top1_base","top1_full"]
for sn in subsets:
    print(f"\n=== {sn} ===")
    print(f"{'指标':<20}"+"".join(f"{k:>14}" for k in res))
    for m in keys: print(f"{m:<20}"+"".join(f"{res[k][sn][m]:>14.4f}" for k in res))
    c=next(iter(res.values()))[sn]; print(f"{'(n 整词错/声调错/对)':<20} {c['n_worderr_all']}/{c['n_toneerr_all']}/{c['n_ok']}")
# 说话人级配对 bootstrap: Δ = B − A (有放回抽说话人, 重复的说话人重复计入)
MK=("word_AUC_all","word_AUC_clean","tone_AUC_all","tone_AUC_clean","top1_base")
def boot(A,B,subset):
    rng=np.random.default_rng(0); S=np.array(sorted(set(spk[subset]))); out={m:[] for m in MK}
    ga={s_:np.where((spk==s_)&subset)[0] for s_ in S}
    for _ in range(a.boot):
        ii=np.concatenate([ga[s_] for s_ in rng.choice(S,len(S))])
        ma,mb=metrics(A,ii),metrics(B,ii)
        for m in MK: out[m].append(mb[m]-ma[m])
    return {m:(float(np.nanmean(v)),float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))) for m,v in out.items()}
VARS=[k for k in ARR]
pairs=[(("v1","clean"),("v2","clean"),"v2−v1 (同配方重跑=噪声底)"),
       (("v1","leaky"),("v1","clean"),"无泄漏−有泄漏 (v1)")]
for v in ("v3b_best","v3b_last","v3b_avg5"):
    pairs.append((("v2","clean"),(v,"clean"),f"{v}−v2"))
res["bootstrap"]={}
print(f"\n=== 说话人级 bootstrap 95% CI ({a.boot} 次, orig32 子集, 说话人为重抽单位) ===")
for A,B,desc in pairs:
    if A not in ARR or B not in ARR: continue
    r=boot(ARR[A],ARR[B],isorig); res["bootstrap"][desc]=r
    for m,(mu,lo,hi) in r.items():
        print(f"  {desc:<26} {m:<15} Δ={mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {'显著' if lo>0 or hi<0 else '不显著'}")
json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False); print(f"\n→ {a.out}")
