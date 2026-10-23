#!/usr/bin/env python3
"""GPU 只做一件事: 对每条留出 clip，算它对"候选全集"里每个词的 CTC 负对数似然，原样存盘。
   所有打分(P_word/P_tone/top-1/泄漏对照/新对手集)都在 CPU 上离线从这份 loss 里算。
   候选全集 = 真实词表 + 每个词基×7调 + 送气变体×7调 + 全部 realized 词(超集; 选对手时再按折过滤)
用法: python extract_losses.py --ckpt X --manifest-dir manifest_full33/fold0_rlz \
        --labels-csv manifest_full33/labels/fold0_heldout.csv --index full33_index.csv --out X.npz"""
import argparse,csv,os,time
import numpy as np, torch, torch.nn.functional as F, soundfile as sf
VALID_TONE=list("bsjvgm"); TONES=VALID_TONE+[""]; VOWELS=set("aeiou")
def base_of(w): return w[:-1] if w and w[-1] in VALID_TONE else w
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
    wav,sr=sf.read(p,dtype='float32')
    if wav.ndim>1: wav=wav.mean(-1)
    assert sr==16000, (p,sr)
    return torch.from_numpy(np.ascontiguousarray(wav))
def build_universe(index_csv):
    vocab=set(); extra=set()
    for r in csv.DictReader(open(index_csv)):
        w=(r["word"] or "").strip().lower()
        if w.isalpha(): vocab.add(w)
        rl=(r.get("realized") or "").strip().lower()
        if rl.isalpha(): extra.add(rl)
    cand={}
    for w in sorted(vocab): cand[w]=True
    for b in sorted({base_of(w) for w in vocab}):
        for t in TONES: cand.setdefault(b+t,False)
        for v in sorted(asp_variants(b)):
            for t in TONES: cand.setdefault(v+t,False)
    for w in sorted(extra): cand.setdefault(w,False)
    return cand
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--w2v-path",default="/staging/x/xouyang7/xlsr_300m.pt")
    ap.add_argument("--manifest-dir",required=True); ap.add_argument("--labels-csv",required=True)
    ap.add_argument("--index",default="full33_index.csv"); ap.add_argument("--data-dir",default=".")
    ap.add_argument("--out",required=True); ap.add_argument("--max-n",type=int,default=0)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()
    rows=[r for r in csv.DictReader(open(a.labels_csv)) if r["label"].strip() in ("1","2","3")]
    if a.max_n: rows=rows[:a.max_n]
    cand=build_universe(a.index)
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
    model=task.build_model(cfg.model); missing=model.load_state_dict(state["model"],strict=False)
    print("load_state_dict:",missing,flush=True)
    model.to(a.device).eval()
    tgt=task.target_dictionary
    dict_syms=[tgt[i] for i in range(len(tgt))]
    normalize=bool(getattr(cfg.task,"normalize",False))
    cw=[];ci=[];ir=[]
    for w,real in cand.items():
        idx=[tgt.index(c) for c in w]
        if any(i==tgt.unk() for i in idx): continue
        cw.append(w); ci.append(idx); ir.append(real)
    N=len(cw); Smax=max(len(c) for c in ci)
    targets=torch.zeros(N,Smax,dtype=torch.long); tlen=torch.zeros(N,dtype=torch.long)
    for i,c in enumerate(ci): targets[i,:len(c)]=torch.tensor(c); tlen[i]=len(c)
    targets=targets.to(a.device); tlen=tlen.to(a.device)
    print(f"candidates={N} (real {sum(ir)})  clips={len(rows)}  normalize={normalize}  device={a.device}",flush=True)
    L=np.full((len(rows),N),np.nan,dtype=np.float32); T=np.zeros(len(rows),dtype=np.int32); t0=time.time()
    for k,r in enumerate(rows):
        wav=load_audio(os.path.join(a.data_dir,r["wav_file"]))
        if normalize: wav=F.layer_norm(wav,wav.shape)
        x=wav.unsqueeze(0).to(a.device); pm=torch.zeros(1,x.shape[1],dtype=torch.bool,device=a.device)
        with torch.no_grad():
            o=model(source=x,padding_mask=pm)
            lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
            if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
            emis=F.log_softmax(lg.float(),dim=-1)
            Tn=emis.shape[0]
            lp=emis.unsqueeze(1).expand(Tn,N,emis.shape[1])
            il=torch.full((N,),Tn,dtype=torch.long,device=a.device)
            L[k]=F.ctc_loss(lp,targets,il,tlen,blank=0,reduction='none',zero_infinity=True).cpu().numpy()
        T[k]=Tn
        if (k+1)%1000==0: print(f"  {k+1}/{len(rows)}  {(k+1)/(time.time()-t0):.1f} clip/s",flush=True)
    assert not np.isnan(L).any()
    np.savez(a.out+".part.npz",loss=L,frames=T,cand=np.array(cw),is_real=np.array(ir),
             wav=np.array([r["wav_file"] for r in rows]),word=np.array([r["word"].strip().lower() for r in rows]),
             label=np.array([int(r["label"]) for r in rows]),speaker=np.array([r["speaker"] for r in rows]),
             realized=np.array([(r.get("realized") or "").strip().lower() for r in rows]),
             dict_syms=np.array(dict_syms),ckpt=np.array(a.ckpt))
    os.replace(a.out+".part.npz",a.out)
    print(f"saved {a.out}  {L.shape}  {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
