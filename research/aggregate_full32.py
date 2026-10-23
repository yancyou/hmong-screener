#!/usr/bin/env python3
"""汇总 4 个 fold 的 v1/v2 评估 -> 全说话人 held-out 总报告 + 三档临床工作点"""
import json,glob,collections,numpy as np

def pool(pat):
    out=[]
    for f in sorted(glob.glob(pat)):
        d=json.load(open(f)); out.append((f,d))
    return out

print("═"*62); print("v1 (与 June 对标: base_top1 0.926 / tone FPR 0.460)"); print("═"*62)
v1=pool("fold*_v1.json"); agg=collections.defaultdict(float); n1=0
for f,d in v1:
    print(f"{f}: base_top1={d.get('label1_base_word_top1')}  "
          f"ww_rec={d.get('wrongword_recall_label34')}@FPR{d.get('wrongword_FPR_label1')}  "
          f"tone_rec={d.get('tone_recall_label2')}@FPR{d.get('tone_FPR_label1')}")
    w=d.get('per_label_total',{}).get('1',d.get('per_label_total',{}).get(1,0))
    for k in ('label1_base_word_top1','wrongword_recall_label34','wrongword_FPR_label1',
              'tone_recall_label2','tone_FPR_label1'):
        if d.get(k) is not None: agg[k]+=d[k]*w
    n1+=w
if n1: print("加权汇总:",{k:round(v/n1,4) for k,v in agg.items()})

print(); print("═"*62); print("v2 (混淆先验)"); print("═"*62)
v2=pool("fold*_v2.json"); allscores=[]
for f,d in v2:
    print(f"\n{f}:")
    for k in ("word_detect(P_word)","tone_detect(P_tone)","any_detect(P_joint)"):
        r=d.get(k)
        if r: print(f"  {k}: AUC={r['AUC']}  R@FPR5={r['recall@FPR5']}  R@FPR10={r['recall@FPR10']}")
    allscores+=d.get("clip_scores",[])
print(f"\n合并 clip_scores: {len(allscores)} 条")
lab=np.array([s['label'] for s in allscores])
pj=np.array([s['P_joint'] for s in allscores])
err=(lab!=1)
def roc_pool(sc,pos):
    o=np.argsort(sc); s=sc[o]; p=pos[o]
    P=p.sum(); N=len(p)-P
    best={}
    cum_p=np.cumsum(p); cum_n=np.cumsum(~p)
    for cap in (0.05,0.10):
        idx=np.where(cum_n/N<=cap)[0]
        if len(idx): k=idx[-1]; best[f"R@FPR{int(cap*100)}"]={"thr":round(float(s[k]),4),"recall":round(float(cum_p[k]/P),4),"FPR":round(float(cum_n[k]/N),4)}
    return best
print("汇总 P_joint (任意错误检出):",roc_pool(pj,err))

# 三档临床工作点: auto-flag 精确率>=0.90, auto-pass NPV>=0.99
ths=np.unique(pj)
flag_thr=None
for t in ths:
    m=pj<t
    if m.sum()>=20 and (err[m].mean() if m.sum() else 0)>=0.90: flag_thr=t
flag_thr=flag_thr if flag_thr is not None else float(ths[0])
pass_thr=None
for t in ths[::-1]:
    m=pj>=t
    if m.sum()>=20 and ((~err[m]).mean() if m.sum() else 0)>=0.99: pass_thr=t
pass_thr=pass_thr if pass_thr is not None else float(ths[-1])
fl=pj<flag_thr; ps=pj>=pass_thr; un=~fl&~ps
print(f"\n三档工作点: flag<{flag_thr:.4f}  pass>={pass_thr:.4f}")
print(f"  auto-flag: {fl.mean()*100:.1f}% 样本, 精确率 {err[fl].mean() if fl.sum() else 0:.3f}")
print(f"  auto-pass: {ps.mean()*100:.1f}% 样本, 正确率 {(~err[ps]).mean() if ps.sum() else 0:.3f}")
print(f"  人工复核: {un.mean()*100:.1f}% 样本 (其中真错误 {err[un].sum()}/{int(err.sum())})")

# 分声调对召回
pair=collections.defaultdict(lambda:[0,0])
for _,d in v2:
    for k,v in d.get("tone_pair_recall@0.5",{}).items():
        pair[k][0]+=v["caught"]; pair[k][1]+=v["total"]
print("\n声调混淆对召回 (P_tone<0.5):")
for k,(c,t) in sorted(pair.items(),key=lambda x:-x[1][1]):
    print(f"  {k:<6} {c:>4}/{t:<4} = {c/max(t,1):.2f}")
