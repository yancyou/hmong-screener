#!/usr/bin/env python3
"""离线打分: 读 extract_losses.py 的 loss 矩阵 -> 每条 clip 的 P_word / P_tone / top-1。
   --priors clean  : 每折只用训练侧说话人的混淆先验 (priors33/priors_fold{k}.json)  ← 无泄漏
   --priors leaky  : 旧的全 32 人先验 (confusion_priors.json)，用于量化泄漏
   打分公式与 closed_vocab_verify_v2.py 完全一致。"""
import json,re,collections,os

def loss_path(evals_dir,prefix,fold):
    """Held-out loss file of one model and fold. HMONG_FOLD_FILES="v1:1=v1fix,..." swaps the file prefix for that model
    and fold, e.g. to use the retrained v1 fold-1 model instead of the mislabeled checkpoint."""
    for item in filter(None,os.environ.get("HMONG_FOLD_FILES","").split(",")):
        key,new=item.split("="); p,f=key.split(":")
        if p==prefix and int(f)==fold: prefix=new
    return f"{evals_dir}/{prefix}_fold{fold}_loss.npz"
import numpy as np
VALID_TONE=list("bsjvgm"); TONES=VALID_TONE+[""]; VOWELS=set("aeiou")
def base_of(w): return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w): return w[-1] if w and w[-1] in VALID_TONE else ""
def onset_of(b):
    for i,ch in enumerate(b):
        if ch in VOWELS: return b[:i]
    return b
def rime_of(b):
    for i,ch in enumerate(b):
        if ch in VOWELS: return b[i:]
    return ""
def asp_variants(b):
    o=onset_of(b); r=rime_of(b)
    if not o or not r: return set()
    outs=set()
    for i,ch in enumerate(o):
        if ch=="h": outs.add(o[:i]+o[i+1:]+r)
    outs.add(o.replace("h","")+r)
    for i in range(len(o)):
        if o[i]!="h" and (i+1>=len(o) or o[i+1]!="h"): outs.add(o[:i+1]+"h"+o[i+1:]+r)
    if o and o[0] in "mnl": outs.add("h"+o+r)
    outs.discard(b)
    return {v for v in outs if v and v.isalpha()}
def load_riv(priors_path):
    wr=collections.defaultdict(set)
    for k in json.load(open(priors_path))["word_pairs"]:
        w,r=k.split(">"); wr[w].add(base_of(r))
    return wr
def score_npz(path,priors_path):
    d=np.load(path,allow_pickle=False)
    return score_arrays(d,d["loss"],priors_path)
def deploy_allowed(cw,ir,priors_path):
    """部署端的候选全集 = 真实词 + 词基×7调 + 送气变体×7调 + 先验里出现过的 realized 词。
       评估时用该折训练侧的先验 → 测试说话人自己读成的词不会混进候选。"""
    real_bases={base_of(w) for w,r in zip(cw,ir) if r}
    asp_bases=set().union(*[asp_variants(b) for b in real_bases])
    extra={k.split(">")[1] for k in json.load(open(priors_path))["word_pairs"]}
    return np.array([bool(r) or base_of(w) in real_bases or base_of(w) in asp_bases or w in extra for w,r in zip(cw,ir)])
def score_arrays(d,L,priors_path,deploy_universe=True):
    """d: 提供 cand/is_real/word/wav/label/speaker 的 npz;  L: [clip, cand] loss (可以是多个模型的平均)
       deploy_universe=True: 候选限定为部署端同款全集(默认, 与线上一致且无泄漏)"""
    cw=d["cand"]; ir=d["is_real"]
    if deploy_universe:
        keep=deploy_allowed(cw,ir,priors_path)
        cw=cw[keep]; ir=ir[keep]; L=L[:,keep]
    cb=np.array([base_of(w) for w in cw]); ct=np.array([tone_of(w) for w in cw]); cr=np.array([rime_of(b) for b in cb])
    real_idx=np.where(ir)[0]; real_w=cw[real_idx]
    wr=load_riv(priors_path); cache={}
    out=[]
    for k,word in enumerate(d["word"]):
        if word not in cache:
            tb=base_of(word)
            riv={tb}|wr.get(word,set())|asp_variants(tb)
            msel=np.where(np.isin(cb,list(riv))|((cr==rime_of(tb))&ir))[0]
            tsel=np.where(cb==tb)[0]
            cache[word]=(msel,cb[msel]==tb,tsel,ct[tsel]==tone_of(word))
        msel,mt,tsel,tt=cache[word]
        v=L[k,msel]; e=np.exp(-(v-v.min())); Pw=float(e[mt].sum()/e.sum())
        v=L[k,tsel]; e=np.exp(-(v-v.min())); Pt=float(e[tt].sum()/e.sum())
        top=real_w[np.argmin(L[k,real_idx])]
        out.append(dict(wav=str(d["wav"][k]),word=str(word),label=int(d["label"][k]),spk=str(d["speaker"][k]),
                        P_word=Pw,P_tone=Pt,top1=str(top),n_msel=len(msel)))
    return out

def score_fast(d,L,priors_path,deploy_universe=True):
    """score_arrays 的向量化版本(按目标词分组批量算)，返回与 d["wav"] 对齐的数组:
       Pw, Pt, hit_base(top-1 词基是否等于目标)。公式与 score_arrays 完全相同。"""
    cw=d["cand"]; ir=d["is_real"]
    if deploy_universe:
        keep=deploy_allowed(cw,ir,priors_path); cw=cw[keep]; ir=ir[keep]; L=L[:,keep]
    cb=np.array([base_of(w) for w in cw]); ct=np.array([tone_of(w) for w in cw]); cr=np.array([rime_of(b) for b in cb])
    wr=load_riv(priors_path); words=d["word"]; n=len(words)
    Pw=np.zeros(n); Pt=np.zeros(n)
    real_idx=np.where(ir)[0]; top=np.asarray(cw[real_idx][np.argmin(L[:,real_idx],axis=1)])
    groups=collections.defaultdict(list)
    for i,w in enumerate(words): groups[str(w)].append(i)
    for word,ii in groups.items():
        ii=np.array(ii); tb=base_of(word)
        riv={tb}|wr.get(word,set())|asp_variants(tb)
        msel=np.where(np.isin(cb,list(riv))|((cr==rime_of(tb))&ir))[0]; mt=cb[msel]==tb
        tsel=np.where(cb==tb)[0]; tt=ct[tsel]==tone_of(word)
        v=L[np.ix_(ii,msel)]; e=np.exp(-(v-v.min(1,keepdims=True))); Pw[ii]=e[:,mt].sum(1)/e.sum(1)
        v=L[np.ix_(ii,tsel)]; e=np.exp(-(v-v.min(1,keepdims=True))); Pt[ii]=e[:,tt].sum(1)/e.sum(1)
    hit=np.array([base_of(a)==base_of(str(b)) for a,b in zip(top,words)])
    return Pw,Pt,hit
