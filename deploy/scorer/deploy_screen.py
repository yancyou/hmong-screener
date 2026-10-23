#!/usr/bin/env python3
"""
Hmong 发音筛查 —— 部署推理脚本

临床用法: 受试者照 word list 朗读, 每条录音配一个已知目标词。
模型输出三档结论:
    PASS    自动放行 (正确率 99.5%)  -> 无需人工
    REVIEW  送人工复核, 按可疑度排序 -> 优先看排前面的
    (无 FLAG 档 —— 见 README: 基础率 6.2% 使自动判错的精确率无法达标)

用法:
  单条:  python deploy_screen.py --ckpt <final.pt> --wav a.wav --word tsheb
  批量:  python deploy_screen.py --ckpt <final.pt> --csv clips.csv --out result.csv
         (csv 需含 wav_file,word 两列)

输出 CSV 列: wav_file, word, P_word, P_tone, P_joint, verdict, priority
"""
from __future__ import annotations
import argparse,csv,json,os,sys
import numpy as np, torch, torch.nn.functional as F, soundfile as sf

VALID_TONE=list("bsjvgm"); TONES=VALID_TONE+[""]
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
    if sr!=16000:                                   # 与训练数据(build_16k.py)相同的重采样
        from fractions import Fraction; from scipy.signal import resample_poly
        fr=Fraction(16000,sr); wav=resample_poly(wav,fr.numerator,fr.denominator)
    return torch.from_numpy(np.asarray(wav,dtype=np.float32)).float()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="/staging/x/xouyang7/full32_ckpts/full32_foldfinal_infer.pt")
    ap.add_argument("--w2v-path",default="/staging/x/xouyang7/xlsr_300m.pt")
    ap.add_argument("--manifest-dir",default="deploy/assets")
    ap.add_argument("--vocab",default="data/full33_index.csv",help="词表来源(含 word 列)")
    ap.add_argument("--priors",default="deploy/assets/priors_final.json")
    ap.add_argument("--thresholds",default="deploy/assets/deploy_thresholds.json")
    ap.add_argument("--wav"); ap.add_argument("--word")
    ap.add_argument("--csv"); ap.add_argument("--data-dir",default=".")
    ap.add_argument("--out",default="screen_result.csv")
    ap.add_argument("--level",default="0.995",choices=["0.995","0.99"],
                    help="auto-pass 正确率目标: 0.995(保守,放行58%) 或 0.99(放行69%)")
    ap.add_argument("--f0-clf",default="deploy/assets/f0_tone_clf.npz",
                    help="F0 声调分类器；存在则启用 CTC+F0 融合(声调 AUC 0.915->0.939)")
    ap.add_argument("--f0-weight",type=float,default=0.4)
    ap.add_argument("--no-f0",action="store_true",help="禁用 F0 融合")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()

    TH=json.load(open(a.thresholds))
    thr_pass=TH["P_joint"][f"pass@{a.level}"]["thr"]
    pri=json.load(open(a.priors))
    word_riv={}
    for k,c in pri["word_pairs"].items():
        w,r=k.split(">"); word_riv.setdefault(w,set()).add(base_of(r))

    # 待处理条目
    items=[]
    if a.wav and a.word: items=[(a.wav,a.word.strip().lower())]
    elif a.csv:
        for r in csv.DictReader(open(a.csv)):
            items.append((r["wav_file"],(r["word"] or "").strip().lower()))
    else: sys.exit("需要 --wav+--word 或 --csv")

    vocab=set()
    for r in csv.DictReader(open(a.vocab)):
        w=(r.get("word") or "").strip().lower()
        if w and w.isalpha(): vocab.add(w)
    vocab=sorted(vocab); real_bases=sorted({base_of(w) for w in vocab})

    from fairseq import tasks, checkpoint_utils
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

    # 候选集: 真实词 + 全声调变体 + 送气变体 + 先验里实测读成的词 (与 ctc_scorer / 离线评估完全相同)
    cand={w:True for w in vocab}
    for b in real_bases:
        for t in TONES: cand.setdefault(b+t,False)
        for v in asp_variants(b):
            for t in TONES: cand.setdefault(v+t,False)
    for k in (json.load(open(a.priors)).get("word_pairs",{}) if os.path.exists(a.priors) else {}):
        rv=k.split(">")[1]
        if rv.isalpha(): cand.setdefault(rv,False)
    cw=[];ci=[]
    for w in cand:
        idx=[tgt.index(c) for c in w]
        if idx and all(i!=tgt.unk() for i in idx): cw.append(w); ci.append(idx)
    N=len(cw); Smax=max(len(c) for c in ci)
    targets=torch.zeros(N,Smax,dtype=torch.long); tlen=torch.zeros(N,dtype=torch.long)
    for i,c in enumerate(ci): targets[i,:len(c)]=torch.tensor(c); tlen[i]=len(c)
    targets=targets.to(a.device); tlen=tlen.to(a.device)
    cb=np.array([base_of(w) for w in cw]); ct=np.array([tone_of(w) for w in cw])
    cr=np.array([rime_of(b) for b in cb]); ir=np.array([cand[w] for w in cw])
    F0S=None
    if not a.no_f0 and os.path.exists(a.f0_clf):
        try:
            from f0_infer import F0ToneScorer
            F0S=F0ToneScorer(a.f0_clf)
            print(f"[deploy] F0 融合已启用 (权重 {a.f0_weight})",flush=True)
        except Exception as e:
            print(f"[deploy] F0 融合不可用: {e}",flush=True)
    print(f"[deploy] 候选 {N} | 待测 {len(items)} | 阈值 P_joint>={thr_pass:.5g} (放行正确率 {a.level})",flush=True)

    def emissions(wav):
        if normalize: wav=F.layer_norm(wav,wav.shape)
        x=wav.unsqueeze(0).to(a.device); pm=torch.zeros(1,x.shape[1],dtype=torch.bool,device=a.device)
        with torch.no_grad():
            o=model(source=x,padding_mask=pm)
            lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
        if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
        return F.log_softmax(lg.float(),dim=-1)
    def nll(emis,idx=None):
        """只对需要的候选算 CTC（目标词已知 -> 无需全表打分，省 ~97% 计算）"""
        T=emis.shape[0]
        tg = targets if idx is None else targets[idx]
        tl = tlen    if idx is None else tlen[idx]
        M=tg.shape[0]
        lp=emis.unsqueeze(1).expand(T,M,emis.shape[1])
        il=torch.full((M,),T,dtype=torch.long,device=a.device)
        return F.ctc_loss(lp,tg,il,tl,blank=blank,reduction='none',zero_infinity=True).cpu().numpy()
    sm=lambda v: np.exp(-(v-v.min()))/np.exp(-(v-v.min())).sum()

    res=[]
    for k,(wav,word) in enumerate(items):
        p=os.path.join(a.data_dir,wav)
        if not os.path.exists(p): print(f"  跳过(缺文件) {wav}"); continue
        tb,tt=base_of(word),tone_of(word)
        riv={tb}|word_riv.get(word,set())|asp_variants(tb)
        msel=np.isin(cb,list(riv))|((cr==rime_of(tb))&ir)
        tsel=(cb==tb)
        need=np.where(msel|tsel)[0]                     # 只算真正用到的候选
        sub=torch.from_numpy(need).to(a.device)
        wav16=load_audio(p)
        loss_sub=nll(emissions(wav16),sub)
        pos={g:k for k,g in enumerate(need)}
        mi=np.array([pos[g] for g in np.where(msel)[0]])
        ti=np.array([pos[g] for g in np.where(tsel)[0]])
        lw=sm(loss_sub[mi]); P_word=float(lw[cb[msel]==tb].sum())
        lt=sm(loss_sub[ti]); tn=ct[tsel]
        P_tone=float(lt[tn==tt].sum()) if tsel.any() else 0.0
        if k==0: print(f"  [优化] 候选 {N} -> 实际计算 {len(need)} ({len(need)/N*100:.1f}%)",flush=True)
        P_joint=P_word*P_tone
        verdict="PASS" if P_joint>=thr_pass else "REVIEW"
        row={"wav_file":wav,"word":word,"P_word":round(P_word,5),
             "P_tone":round(P_tone,5),"P_joint":round(P_joint,6),"verdict":verdict}
        if F0S is not None:
            try:                                        # F0 与训练特征一致: 16 kHz 音频上跑 pYIN
                row["_f0raw"]=F0S.raw_logf0(wav16.numpy(),16000)   # 先存原始音高; 整场收齐后做说话人级归一化
            except Exception:
                row["_f0raw"]=None
        res.append(row)
        if (k+1)%200==0: print(f"  {k+1}/{len(items)}",flush=True)
    if F0S is not None:
        # 说话人级归一化(与 F0 分类器训练特征一致): 用本场全部录音估计音高基准
        F0S.fit_speaker([r["_f0raw"][0] for r in res if r.get("_f0raw")],speaker="_session")
        for r in res:
            fr=r.pop("_f0raw",None)
            r["P_f0_tone"]=round(float(F0S.posterior_from_logf0(*fr,r["word"],speaker="_session")[0]),5) if fr else None
    if F0S is not None and any(r.get("P_f0_tone") is not None for r in res) and len(res)>=10:
        import numpy as _np
        from scipy.stats import rankdata as _rk
        ok=[r for r in res if r.get("P_f0_tone") is not None]
        rt=_rk([r["P_tone"] for r in ok])/len(ok)
        rf=_rk([r["P_f0_tone"] for r in ok])/len(ok)
        rw=_rk([r["P_word"] for r in ok])/len(ok)
        w=a.f0_weight
        for i,r in enumerate(ok):
            r["tone_fused"]=round(float((1-w)*rt[i]+w*rf[i]),5)
            r["suspicion"]=round(float(min(rw[i],r["tone_fused"])),5)   # 越低越可疑
        # 会话内按 suspicion 定档：保持与全局阈值等价的放行比例
        n_pass=sum(1 for r in ok if r["P_joint"]>=thr_pass)
        for i,r in enumerate(sorted(ok,key=lambda r:-r["suspicion"])):
            r["verdict"]="PASS" if i<n_pass else "REVIEW"
        res.sort(key=lambda r:(r.get("suspicion") if r.get("suspicion") is not None else r["P_joint"]))
    else:
        res.sort(key=lambda r:r["P_joint"])
    for i,r in enumerate(res): r["priority"]=i+1 if r["verdict"]=="REVIEW" else ""
    with open(a.out,'w',newline='') as f:
        cols=["wav_file","word","P_word","P_tone","P_f0_tone","tone_fused","suspicion","P_joint","verdict","priority"]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader()
        for r in res: w.writerow(r)
    np_=sum(1 for r in res if r["verdict"]=="PASS")
    print(f"\n完成: {len(res)} 条 -> {a.out}")
    print(f"  PASS   {np_} ({np_/max(len(res),1)*100:.1f}%)  自动放行")
    print(f"  REVIEW {len(res)-np_} ({(len(res)-np_)/max(len(res),1)*100:.1f}%)  按 priority 排序人工复核")
    if len(items)==1 and res:
        r=res[0]; print(f"\n  {r['word']}: P_word={r['P_word']} P_tone={r['P_tone']} -> {r['verdict']}")

if __name__=="__main__": main()
