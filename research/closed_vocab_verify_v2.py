#!/usr/bin/env python3
"""
CLOSED-VOCAB verification v2: 在 v1 (CTC 闭集重打分) 基础上加入 full32 实测混淆先验。
升级点:
  1) 候选集 = 真实词表 + 全部声调变体(每个 base × 7调) + 送气变体 + 实测 realized 词
  2) 词判定: 目标 vs {实测词对对手 ∪ 送气变体 ∪ 同韵 cohort} 的局部 softmax -> P_word
  3) 声调判定: 目标 base 的 7 调局部后验 -> p_tone; 只把质量流向"实测混淆调"计为风险
  4) 输出: 词/调/联合三条 ROC (recall@FPR5/10) + 按混淆对分解的召回表
用法:
  python closed_vocab_verify_v2.py --ckpt <fold_best.pt> --w2v-path /staging/x/xouyang7/xlsr_300m.pt \
     --manifest-dir manifest_full32/fold0_rlz --labels-csv manifest_full32/labels/fold0_heldout.csv \
     --index full32_index.csv --priors confusion_priors.json --data-dir . --out fold0_v2.json
"""
from __future__ import annotations
import argparse,csv,json,os,collections
import numpy as np
import torch, torch.nn.functional as F
import soundfile as sf

VALID_TONE=list("bsjvgm"); TONES=VALID_TONE+[""]   # "" = 中平调(无字母)
VOWELS=set("aeiou")
def base_of(w): w=(w or "").strip().lower(); return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w): w=(w or "").strip().lower(); return w[-1] if w and w[-1] in VALID_TONE else ""
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
def load_audio(p):
    wav,sr=sf.read(p)
    if wav.ndim>1: wav=wav.mean(-1)
    if sr!=16000:
        import librosa; wav=librosa.resample(np.asarray(wav,dtype=np.float32),orig_sr=sr,target_sr=16000)
    return torch.from_numpy(np.asarray(wav,dtype=np.float32)).float()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--w2v-path",default="/staging/x/xouyang7/xlsr_300m.pt")
    ap.add_argument("--manifest-dir",required=True); ap.add_argument("--labels-csv",required=True)
    ap.add_argument("--index",default="full32_index.csv"); ap.add_argument("--priors",default="confusion_priors.json")
    ap.add_argument("--data-dir",default="."); ap.add_argument("--out",required=True)
    ap.add_argument("--min-pair-count",type=int,default=3)
    ap.add_argument("--max-n",type=int,default=0)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()

    pri=json.load(open(a.priors))
    tone_next=collections.defaultdict(set)     # 实测: 目标调 -> 混淆调集合
    for k,c in pri["tone_pairs"].items():
        if c>=a.min_pair_count:
            x,y=k.split(">"); tone_next["" if x=="n" else x].add("" if y=="n" else y)
    word_riv=collections.defaultdict(set)      # 实测: 词 -> 对手 base 集合
    for k,c in pri["word_pairs"].items():
        w,r=k.split(">"); word_riv[w].add(base_of(r))

    vocab=set(); extra=set()
    for r in csv.DictReader(open(a.index)):
        w=(r["word"] or "").strip().lower()
        if w: vocab.add(w)
        rl=(r.get("realized") or "").strip().lower()
        if rl: extra.add(rl)
    rows=[]
    for r in csv.DictReader(open(a.labels_csv)):
        lbl=r.get("label","").strip()
        if lbl not in ("1","2","3","4"): continue
        rows.append((r["wav_file"],r["word"].strip().lower(),int(lbl),r["speaker"].strip().lower(),
                     (r.get("realized") or "").strip().lower()))
    if a.max_n and len(rows)>a.max_n:
        import random; random.Random(0).shuffle(rows); rows=rows[:a.max_n]
    vocab=sorted(vocab)
    real_bases=sorted({base_of(w) for w in vocab})
    cand=dict()                                 # word -> is_real
    for w in vocab: cand[w]=True
    for b in real_bases:
        for t in TONES: cand.setdefault(b+t,False)      # 全部声调变体
        for v in asp_variants(b):
            for t in TONES: cand.setdefault(v+t,False)  # 送气变体×声调
    for w in extra: cand.setdefault(w,False)
    from fairseq import tasks,checkpoint_utils
    ov={"data":os.path.abspath(a.manifest_dir)}
    if os.path.exists(a.w2v_path): ov["w2v_path"]=os.path.abspath(a.w2v_path)
    state=checkpoint_utils.load_checkpoint_to_cpu(a.ckpt,arg_overrides={"model":ov})
    cfg=state["cfg"]
    try: cfg.model.w2v_path=os.path.abspath(a.w2v_path)
    except Exception: pass
    mdir=os.path.abspath(a.manifest_dir)
    for at in ("data","label_dir"):
        try:
            if getattr(cfg.task,at,None) is not None: setattr(cfg.task,at,mdir)
        except Exception: pass
    task=tasks.setup_task(cfg.task)
    model=task.build_model(cfg.model); model.load_state_dict(state["model"],strict=False)
    model.to(a.device).eval()
    tgt=task.target_dictionary; blank=0
    normalize=bool(getattr(getattr(cfg,"task",object()),"normalize",False))

    def to_idx(w): return [tgt.index(c) for c in w if tgt.index(c)!=tgt.unk()]
    cw=[];ci=[];ir=[]
    for w,real in cand.items():
        idx=to_idx(w)
        if idx: cw.append(w); ci.append(idx); ir.append(real)
    N=len(cw); Smax=max(len(c) for c in ci)
    targets=torch.zeros(N,Smax,dtype=torch.long); tlen=torch.zeros(N,dtype=torch.long)
    for i,c in enumerate(ci): targets[i,:len(c)]=torch.tensor(c); tlen[i]=len(c)
    targets=targets.to(a.device); tlen=tlen.to(a.device)
    cb=np.array([base_of(w) for w in cw]); ct=np.array([tone_of(w) for w in cw])
    cr=np.array([rime_of(b) for b in cb]); ir=np.array(ir)
    print(f"[v2] candidates={N} (real {int(ir.sum())})  test clips={len(rows)}")

    def emissions(wav):
        if normalize: wav=F.layer_norm(wav,wav.shape)
        x=wav.unsqueeze(0).to(a.device); pm=torch.zeros(1,x.shape[1],dtype=torch.bool,device=a.device)
        with torch.no_grad():
            o=model(source=x,padding_mask=pm)
            lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
        if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
        return F.log_softmax(lg.float(),dim=-1)
    def nll_all(emis):
        T=emis.shape[0]
        lp=emis.unsqueeze(1).expand(T,N,emis.shape[1])
        il=torch.full((N,),T,dtype=torch.long,device=a.device)
        return F.ctc_loss(lp,targets,il,tlen,blank=blank,reduction='none',zero_infinity=True).cpu().numpy()
    sm=lambda v: np.exp(-(v-v.min()))/np.exp(-(v-v.min())).sum()

    scores=[]; miss=0; pair_rec=collections.defaultdict(lambda:[0,0])
    for k,(wav,word,lbl,spk,realized) in enumerate(rows):
        p=os.path.join(a.data_dir,wav)
        if not os.path.exists(p): miss+=1; continue
        try: loss=nll_all(emissions(load_audio(p)))
        except Exception: miss+=1; continue
        tb,tt=base_of(word),tone_of(word)
        # --- 词: 局部对手 ---
        riv=({tb}|word_riv.get(word,set())|asp_variants(tb))
        msel=np.isin(cb,list(riv))|((cr==rime_of(tb))&ir)
        lw=sm(loss[msel]); P_word=float(lw[cb[msel]==tb].sum())
        # --- 调: 目标 base 的调后验 ---
        tsel=(cb==tb)
        lt=sm(loss[tsel]); tn=ct[tsel]
        p_tone={t:float(lt[tn==t].sum()) for t in TONES}
        P_tt=p_tone.get(tt,0.0)
        attested=float(sum(p_tone.get(t,0.0) for t in tone_next.get(tt,set())))
        P_joint=P_word*P_tt
        scores.append({"label":lbl,"spk":spk,"word":word,"P_word":round(P_word,5),
                       "P_tone":round(P_tt,5),"P_att_riv":round(attested,5),"P_joint":round(P_joint,5)})
        if lbl==2 and realized:
            pair_rec[f"{tt or 'n'}>{tone_of(realized) or 'n'}"][1]+=1
            if P_tt<0.5: pair_rec[f"{tt or 'n'}>{tone_of(realized) or 'n'}"][0]+=1
        if (k+1)%500==0: print(f"  {k+1}/{len(rows)}")

    def roc(key,pos_lbls):
        pos=[s[key] for s in scores if s["label"] in pos_lbls]
        neg=[s[key] for s in scores if s["label"]==1]
        if not pos or not neg: return None
        def at_fpr(cap):
            best=None
            for thr in sorted(set(pos+neg)):
                fpr=sum(1 for x in neg if x<thr)/len(neg)
                if fpr<=cap:
                    rec=sum(1 for x in pos if x<thr)/len(pos)
                    if best is None or rec>best[1]: best=(thr,rec,fpr)
            return None if best is None else {"thr":round(best[0],5),"recall":round(best[1],4),"FPR":round(best[2],4)}
        auc=0.0
        sp=sorted(pos); sn=sorted(neg)
        import bisect
        for x in sp: auc+=bisect.bisect_left(sn,x)+0.5*(bisect.bisect_right(sn,x)-bisect.bisect_left(sn,x))
        auc=1-auc/(len(sp)*len(sn))
        return {"n_err":len(pos),"n_ok":len(neg),"AUC":round(auc,4),
                "recall@FPR5":at_fpr(0.05),"recall@FPR10":at_fpr(0.10)}
    out={"n_test":len(rows),"n_miss":miss,"candidates":N,
         "word_detect(P_word)":roc("P_word",(3,4)),
         "tone_detect(P_tone)":roc("P_tone",(2,)),
         "any_detect(P_joint)":roc("P_joint",(2,3,4)),
         "tone_pair_recall@0.5":{k:{"caught":v[0],"total":v[1]} for k,v in sorted(pair_rec.items())},
         "clip_scores":scores}
    json.dump(out,open(a.out,"w"),indent=1)
    for k in ("word_detect(P_word)","tone_detect(P_tone)","any_detect(P_joint)"):
        print(k,"->",out[k] and {x:out[k][x] for x in ("AUC","recall@FPR5","recall@FPR10")})
if __name__=="__main__": main()
