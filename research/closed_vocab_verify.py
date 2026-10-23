#!/usr/bin/env python3
"""
CLOSED-VOCAB verification POC: rescore each production against the fixed word list
using the ASR model's CTC emissions, instead of free decoding.

Clinical task = read a known word from a fixed list; we always know the target word.
So we don't need open ASR — we pick the best-matching word among the ~1120 known words
(CTC log-likelihood), which is far easier than free decoding on hard audio.

Leave-speaker-out: model trained on 11 speakers; we test held-out speakers.
Reports (on held-out):
  - full-word top-1 / top-5 acc (label-1)        : can we identify the exact word+tone?
  - base-word top-1 / top-5 acc (label-1)        : can we identify the word (ignore tone)?
  - wrong-word (label 3/4) detection             : target-base != top-1-base  -> recall, +FPR on label-1
  - tone (label-2) detection                     : base top-1 right but tone letter wrong -> recall, +FPR
Compares closed-vocab accuracy to free-decode (denoised ASR ~22% CER in-dist).

Usage:
  python closed_vocab_verify.py --ckpt /staging/x/xouyang7/lso_dn_asr_best.pt \
     --w2v-path /staging/x/xouyang7/xlsr_300m.pt --manifest-dir manifest_lso_dn \
     --labels-csv template_manifest_all_xlsx_speakers.csv --data-dir . \
     --only-speakers <spk1>,<spk2>,<spk3> --out closed_vocab_heldout.json
"""
from __future__ import annotations
import argparse, csv, json, os, zlib
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

VALID_TONE = set("bsjvgm")
VOWELS = set("aeiou")
def base_of(w):
    w=(w or "").strip().lower(); return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w):
    w=(w or "").strip().lower(); return w[-1] if w and w[-1] in VALID_TONE else "-"
def onset_of(b):
    for i,ch in enumerate(b):
        if ch in VOWELS: return b[:i]
    return b
def rime_of(b):
    for i,ch in enumerate(b):
        if ch in VOWELS: return b[i:]
    return ""   # no vowel -> empty rime (rare); cohort by full word

def asp_variants(b):
    """Phonetic minimal-pair competitors of base word b: toggle aspiration 'h'
    in the onset (covers the dominant aspiration & pre-aspirated-nasal confusions).
    Returns a set of variant base words (may include non-words; fine for CTC scoring)."""
    o=onset_of(b); r=rime_of(b)
    if not o or not r: return set()
    outs=set()
    # remove each 'h' in onset (one at a time, and all)
    for i,ch in enumerate(o):
        if ch=="h": outs.add(o[:i]+o[i+1:]+r)
    outs.add(o.replace("h","")+r)
    # insert 'h' after each non-h consonant in onset (aspirate a stop/affricate)
    for i in range(len(o)):
        if o[i]!="h" and (i+1>=len(o) or o[i+1]!="h"):
            outs.add(o[:i+1]+"h"+o[i+1:]+r)
    # pre-aspiration on nasals/l: toggle leading h (hm<->m, hn<->n, hl<->l, hny<->ny)
    if o and o[0] in "mnl":
        outs.add("h"+o+r)
    outs.discard(b)
    return {v for v in outs if v and all(c.isalpha() for c in v)}

def load_audio(p):
    wav,sr=sf.read(p)
    if wav.ndim>1: wav=wav.mean(-1)
    if sr!=16000:
        import librosa; wav=librosa.resample(np.asarray(wav,dtype=np.float32),orig_sr=sr,target_sr=16000)
    return torch.from_numpy(np.asarray(wav,dtype=np.float32)).float()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--w2v-path",default="/staging/x/xouyang7/xlsr_300m.pt")
    ap.add_argument("--manifest-dir",default="manifest_lso_dn")
    ap.add_argument("--labels-csv",required=True)
    ap.add_argument("--data-dir",default=".")
    ap.add_argument("--only-speakers",default="")
    ap.add_argument("--exclude-speakers",default="")
    ap.add_argument("--max-n",type=int,default=0)
    ap.add_argument("--label1-crc-rem",type=int,default=-1,
                    help="if >=0, for label-1 keep only clips where crc32(basename)%%2==rem (enrollment test-split, no leakage)")
    ap.add_argument("--out",required=True)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()
    only_spk={s.strip().lower() for s in a.only_speakers.split(",") if s.strip()}
    excl_spk={s.strip().lower() for s in a.exclude_speakers.split(",") if s.strip()}

    from fairseq import tasks, checkpoint_utils
    ov={"data":os.path.abspath(a.manifest_dir)}
    if os.path.exists(a.w2v_path): ov["w2v_path"]=os.path.abspath(a.w2v_path)
    state=checkpoint_utils.load_checkpoint_to_cpu(a.ckpt,arg_overrides={"model":ov})
    cfg=state["cfg"]
    if os.path.exists(a.w2v_path):
        try: cfg.model.w2v_path=os.path.abspath(a.w2v_path)
        except: pass
    mdir=os.path.abspath(a.manifest_dir)
    for at in ("data","label_dir"):
        try:
            if getattr(cfg.task,at,None) is not None: setattr(cfg.task,at,mdir)
        except: pass
    task=tasks.setup_task(cfg.task)
    model=task.build_model(cfg.model); model.load_state_dict(state["model"],strict=False)
    model.to(a.device).eval()
    tgt=task.target_dictionary; blank=0
    normalize=bool(getattr(getattr(cfg,"task",object()),"normalize",False))
    print(f"[cfg] dict={len(tgt.symbols)} normalize={normalize}")

    # ---- build closed vocabulary (unique words seen in CSV) ----
    vocab=set()
    rows=[]
    for r in csv.DictReader(open(a.labels_csv)):
        w=(r.get("word") or "").strip().lower(); lbl=(r.get("label") or "").strip()
        spk=(r.get("speaker") or "").strip().lower(); wav=(r.get("wav_file") or "").strip()
        if lbl not in ("1","2","3","4") or not w or not wav: continue
        vocab.add(w)
        if only_spk and spk not in only_spk: continue
        if excl_spk and spk in excl_spk: continue
        if a.label1_crc_rem>=0 and lbl=="1" and (zlib.crc32(os.path.basename(wav).encode())%2)!=a.label1_crc_rem:
            continue   # keep only the enrollment TEST half of label-1 (enroll used rem 0)
        rows.append((wav,w,int(lbl),spk))
    if a.max_n and len(rows)>a.max_n:
        import random; random.Random(0).shuffle(rows); rows=rows[:a.max_n]
    vocab=sorted(vocab)
    # ---- contrastive candidates: real vocab + virtual aspiration-toggle variants ----
    # detection pits the target against its same-rime cohort + aspiration variants,
    # so minimal-pair errors (cub->chub) lose probability mass to the rival onset.
    real_bases={base_of(w) for w in vocab}
    virtual=set()
    for b in real_bases:
        for v in asp_variants(b):
            if v not in real_bases: virtual.add(v)
    cand_words=list(vocab)+sorted(virtual)        # virtual entries are toneless base words
    is_real=[True]*len(vocab)+[False]*len(virtual) # only real vocab counts for word-ID top-k
    print(f"contrastive candidates: {len(vocab)} real + {len(virtual)} virtual aspiration variants")
    # candidate index tensors (letters only, no word-boundary)
    def to_idx(w):
        return [tgt.index(c) for c in w if tgt.index(c)!=tgt.unk()]
    cand_idx=[to_idx(w) for w in cand_words]
    keep=[i for i,c in enumerate(cand_idx) if len(c)>0]
    cand_words=[cand_words[i] for i in keep]; cand_idx=[cand_idx[i] for i in keep]
    is_real=[is_real[i] for i in keep]
    vocab=[w for w,r in zip(cand_words,is_real) if r]   # real-vocab list (word-ID metrics)
    Smax=max(len(c) for c in cand_idx)
    targets=torch.zeros(len(cand_idx),Smax,dtype=torch.long)
    tlen=torch.zeros(len(cand_idx),dtype=torch.long)
    for i,c in enumerate(cand_idx):
        targets[i,:len(c)]=torch.tensor(c); tlen[i]=len(c)
    targets=targets.to(a.device); tlen=tlen.to(a.device)
    import numpy as _np
    cw_base=_np.array([base_of(w) for w in cand_words])     # base of every candidate (real+virtual)
    cw_rime=_np.array([rime_of(b) for b in cw_base])        # rime of every candidate
    real_mask=_np.array(is_real)                            # True for real-vocab candidates
    real_pos=_np.where(real_mask)[0]                        # indices of real candidates in cand_words
    vbase=[base_of(w) for w in vocab]                       # base of real vocab (word-ID metrics)
    vbase_arr=_np.array(vbase)
    print(f"closed vocab: {len(vocab)} words, Smax={Smax}; test clips: {len(rows)}")

    def emissions(wav):
        if normalize: wav=F.layer_norm(wav,wav.shape)
        x=wav.unsqueeze(0).to(a.device); pm=torch.zeros(1,x.shape[1],dtype=torch.bool,device=a.device)
        with torch.no_grad():
            o=model(source=x,padding_mask=pm)
            lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
        if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
        return F.log_softmax(lg.float(),dim=-1)  # [T,V]

    N=len(cand_words)   # CTC over ALL candidates (real + virtual aspiration variants)
    def rank(emis):
        T=emis.shape[0]
        lp=emis.unsqueeze(1).expand(T,N,emis.shape[1])  # [T,N,V]
        il=torch.full((N,),T,dtype=torch.long,device=a.device)
        loss=F.ctc_loss(lp,targets,il,tlen,blank=blank,reduction='none',zero_infinity=True)  # [N] nll
        return loss.cpu().numpy()   # full per-candidate NLL

    from collections import defaultdict
    tot=defaultdict(int)
    dir_b=defaultdict(lambda:[0,0]); spk_b=defaultdict(lambda:[0,0])  # label-1 base-top1 [correct,total]
    full_top1=full_top5=base_top1=base_top5=0; n1=0
    wrongword_caught=0; n_ww=0; l1_baseflag=0
    tone_caught=0; n_tone=0; l1_toneflag=0; base_ok_for_tone=0
    miss=0; ex=[]; clip_scores=[]
    softmax_np=lambda v: _np.exp(v-v.max())/_np.exp(v-v.max()).sum()
    for k,(wav,word,lbl,spk) in enumerate(rows):
        p=os.path.join(a.data_dir,wav)
        if not os.path.exists(p): miss+=1; continue
        try:
            loss=rank(emissions(load_audio(p)))   # [N_cand] nll over real+virtual
        except Exception as e:
            miss+=1
            if miss<=3: print("  err",repr(e)[:80]); continue
            continue
        bt0=base_of(word)
        # --- word-ID over REAL vocab only (top-k) ---
        real_loss=loss[real_pos]
        order=_np.argsort(real_loss)
        wp=softmax_np(-real_loss)                       # P(word) over real vocab (global)
        p_tgt=float(wp[vbase_arr==bt0].sum())           # global P(target base)
        # --- contrastive: target vs its same-rime cohort + aspiration variants ---
        rt=rime_of(bt0)
        cohort=_np.where(cw_rime==rt)[0]
        if cohort.size>=2:
            cl=softmax_np(-loss[cohort])                # local softmax within rime cohort
            p_local=float(cl[cw_base[cohort]==bt0].sum())
        else:
            p_local=p_tgt                               # no rivals -> fall back to global
        clip_scores.append({"label":lbl,"spk":spk,
                            "P_target_base":round(p_tgt,5),
                            "P_local_target":round(p_local,5)})
        top=[vocab[i] for i in order[:5]]; topb=[vbase[i] for i in order[:5]]
        bt=base_of(word); tt=tone_of(word)
        tot[lbl]+=1
        if lbl==1:
            n1+=1
            if top[0]==word: full_top1+=1
            if word in top: full_top5+=1
            sdir=wav.split("/")[0]; dir_b[sdir][1]+=1; spk_b[spk][1]+=1
            if topb[0]==bt:
                base_top1+=1; dir_b[sdir][0]+=1; spk_b[spk][0]+=1
            if bt in topb: base_top5+=1
            if topb[0]!=bt: l1_baseflag+=1                 # base-FPR
            if topb[0]==bt:
                base_ok_for_tone+=1
                if tone_of(top[0])!=tt: l1_toneflag+=1     # tone-FPR among base-correct label-1
            if len(ex)<15: ex.append({"target":word,"top5":top})
        if lbl in (3,4):
            n_ww+=1
            if topb[0]!=bt: wrongword_caught+=1            # predicted base != target base -> caught
        if lbl==2:
            n_tone+=1
            # tone error: base should match target; flag if tone letter of top-1 != target tone
            if topb[0]==bt and tone_of(top[0])!=tt: tone_caught+=1
            elif topb[0]!=bt: tone_caught+=1               # also caught (as some mismatch)
        if (k+1)%500==0: print(f"  {k+1}/{len(rows)}  base_top1={base_top1/max(n1,1):.3f}")

    # ---- detection ROC: global P(target) vs contrastive local P(target) ----
    # error = label 2/3/4 (flag if score < thr); FPR = label-1 flagged.
    def roc(key):
        pos=[c[key] for c in clip_scores if c["label"] in (2,3,4)]   # errors
        neg=[c[key] for c in clip_scores if c["label"]==1]           # correct
        if not pos or not neg: return None
        rows_=[]
        for thr in [0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            rec=sum(1 for s in pos if s<thr)/len(pos)
            fpr=sum(1 for s in neg if s<thr)/len(neg)
            rows_.append({"thr":thr,"recall":round(rec,4),"FPR":round(fpr,4)})
        # FPR<=0.05 and <=0.10 operating points (max recall under FPR cap)
        def at_fpr(cap):
            best=None
            for thr in sorted(set(neg+pos)):
                fpr=sum(1 for s in neg if s<thr)/len(neg)
                if fpr<=cap:
                    rec=sum(1 for s in pos if s<thr)/len(pos)
                    if best is None or rec>best[1]: best=(thr,rec,fpr)
            return None if best is None else {"thr":round(best[0],5),"recall":round(best[1],4),"FPR":round(best[2],4)}
        return {"n_err":len(pos),"n_correct":len(neg),"sweep":rows_,
                "recall@FPR5":at_fpr(0.05),"recall@FPR10":at_fpr(0.10)}

    res={
        "n_test":sum(tot.values()),"n_miss":miss,"vocab_size":len(vocab),
        "per_label_total":{str(k):tot[k] for k in sorted(tot)},
        "label1_full_word_top1":round(full_top1/max(n1,1),4),
        "label1_full_word_top5":round(full_top5/max(n1,1),4),
        "label1_base_word_top1":round(base_top1/max(n1,1),4),
        "label1_base_word_top5":round(base_top5/max(n1,1),4),
        "wrongword_recall_label34":round(wrongword_caught/max(n_ww,1),4),
        "wrongword_FPR_label1":round(l1_baseflag/max(n1,1),4),
        "tone_recall_label2":round(tone_caught/max(n_tone,1),4),
        "tone_FPR_label1":round(l1_toneflag/max(base_ok_for_tone,1),4),
        "base_top1_by_dir":{k:round(v[0]/v[1],4) for k,v in sorted(dir_b.items()) if v[1]},
        "base_top1_by_speaker":{k:round(v[0]/v[1],4) for k,v in sorted(spk_b.items()) if v[1]},
        "base_top1_count_by_dir":{k:v[1] for k,v in sorted(dir_b.items())},
        "detect_global_Ptarget":roc("P_target_base"),
        "detect_contrastive_Plocal":roc("P_local_target"),
        "clip_scores":clip_scores,
        "examples":ex,
    }
    print("\n"+"="*64)
    print(f"CLOSED-VOCAB ({len(vocab)} words) on HELD-OUT speakers {sorted(only_spk)}")
    print(f"  label-1 base-word  top1={res['label1_base_word_top1']:.3f}  top5={res['label1_base_word_top5']:.3f}")
    print(f"  label-1 full-word  top1={res['label1_full_word_top1']:.3f}  top5={res['label1_full_word_top5']:.3f}")
    print(f"  wrong-word (3/4): recall={res['wrongword_recall_label34']:.3f}  FPR(label1)={res['wrongword_FPR_label1']:.3f}")
    print(f"  tone (label-2):   recall={res['tone_recall_label2']:.3f}  FPR(label1)={res['tone_FPR_label1']:.3f}")
    for nm,key in [("GLOBAL P(target)","detect_global_Ptarget"),("CONTRASTIVE local P(target)","detect_contrastive_Plocal")]:
        d=res[key]
        if d:
            print(f"  -- {nm}: errors={d['n_err']} correct={d['n_correct']}")
            print(f"       recall@FPR<=5% : {d['recall@FPR5']}")
            print(f"       recall@FPR<=10%: {d['recall@FPR10']}")
    print("  base-top1 BY DIR (audio-quality probe):")
    for k,v in sorted(dir_b.items()):
        if v[1]: print(f"    {k:<22} top1={v[0]/v[1]:.3f}  (n={v[1]})")
    print("  base-top1 BY SPEAKER (low->high):")
    for k,v in sorted(spk_b.items(),key=lambda kv:kv[1][0]/max(kv[1][1],1)):
        if v[1]: print(f"    {k:<12} top1={v[0]/v[1]:.3f}  (n={v[1]})")
    print(f"  n_test={res['n_test']} miss={miss}")
    print("="*64)
    json.dump(res,open(a.out,"w"),indent=2,ensure_ascii=False)
    print("saved",a.out)

if __name__=="__main__":
    main()
