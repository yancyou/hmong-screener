#!/usr/bin/env python3
"""把一条录音的 CTC 中间过程完整打印出来，用于理解推理机制"""
import sys,os,csv,json,numpy as np,torch,torch.nn.functional as F,soundfile as sf
VALID=set("bsjvgm"); V=set("aeiou")
def base_of(w): return w[:-1] if w and w[-1] in VALID else w
def tone_of(w): return w[-1] if w and w[-1] in VALID else ""
def onset(b):
    for i,c in enumerate(b):
        if c in V: return b[:i]
    return b
def rime(b):
    for i,c in enumerate(b):
        if c in V: return b[i:]
    return ""
def asp(b):
    o,r=onset(b),rime(b)
    if not o or not r: return set()
    out=set()
    for i,c in enumerate(o):
        if c=="h": out.add(o[:i]+o[i+1:]+r)
    out.add(o.replace("h","")+r)
    for i in range(len(o)):
        if o[i]!="h" and (i+1>=len(o) or o[i+1]!="h"): out.add(o[:i+1]+"h"+o[i+1:]+r)
    if o and o[0] in "mnl": out.add("h"+o+r)
    out.discard(b); return {v for v in out if v and v.isalpha()}

WAV=sys.argv[1]; WORD=sys.argv[2].strip().lower()
from fairseq import tasks, checkpoint_utils
ov={"data":os.path.abspath("ctc_data")}
st=checkpoint_utils.load_checkpoint_to_cpu("/staging/x/xouyang7/full32_ckpts/full32_foldfinal_best.pt",arg_overrides={"model":ov})
cfg=st["cfg"]
try: cfg.model.w2v_path="/staging/x/xouyang7/xlsr_300m.pt"
except Exception: pass
for at in ("data","label_dir"):
    try:
        if getattr(cfg.task,at,None) is not None: setattr(cfg.task,at,os.path.abspath("ctc_data"))
    except Exception: pass
task=tasks.setup_task(cfg.task); model=task.build_model(cfg.model)
model.load_state_dict(st["model"],strict=False); model.eval()
tgt=task.target_dictionary
norm=bool(getattr(getattr(cfg,"task",object()),"normalize",False))

wav,sr=sf.read(WAV,dtype='float32')
if wav.ndim>1: wav=wav.mean(-1)
x=torch.from_numpy(wav).float()
print(f"录音: {os.path.basename(WAV)}   目标词: {WORD}")
print(f"音频长度 {len(wav)/sr:.2f} 秒 = {len(wav)} 个采样点\n")
if norm: x=F.layer_norm(x,x.shape)
with torch.no_grad():
    o=model(source=x.unsqueeze(0),padding_mask=torch.zeros(1,len(x),dtype=torch.bool))
    lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
emis=F.log_softmax(lg.float(),dim=-1)
T,Vsz=emis.shape
print(f"═══ 模型输出：一张 [{T} 帧 × {Vsz} 个符号] 的概率表 ═══")
print(f"   {len(wav)} 采样点 → {T} 帧（每帧约 {len(wav)/sr/T*1000:.0f} 毫秒）\n")
P=emis.exp().numpy()
syms=[tgt[i] for i in range(Vsz)]
print("每一帧最可能的符号（前 25 帧）:")
print("  帧 | 符号  概率 | 第二可能")
for t in range(min(25,T)):
    o1=np.argsort(-P[t])[:2]
    s1=syms[o1[0]] if syms[o1[0]] not in ('<s>','<pad>') else '<blank>'
    s2=syms[o1[1]] if syms[o1[1]] not in ('<s>','<pad>') else '<blank>'
    print(f"  {t:>3} | {s1:<7} {P[t,o1[0]]:.3f} | {s2:<7} {P[t,o1[1]]:.3f}")
print(f"  ... 共 {T} 帧\n")

# 候选集
vocab=set()
for r in csv.DictReader(open('full32_index.csv')):
    w=(r.get('word') or '').strip().lower()
    if w and w.isalpha(): vocab.add(w)
tb,tt=base_of(WORD),tone_of(WORD)
TONES=list("bsjvgm")+[""]
pri=json.load(open('confusion_priors.json'))
riv={base_of(k.split(">")[1]) for k in pri["word_pairs"] if k.split(">")[0]==WORD}
cands=sorted(set([tb+t for t in TONES])|{v+t for v in asp(tb) for t in TONES}|
             {w for w in vocab if rime(base_of(w))==rime(tb)}|
             {r+t for r in riv for t in TONES})
def to_idx(w): return [tgt.index(c) for c in w if tgt.index(c)!=tgt.unk()]
cw=[];ci=[]
for w in cands:
    ix=to_idx(w)
    if ix: cw.append(w); ci.append(ix)
N=len(cw); S=max(len(c) for c in ci)
tt_=torch.zeros(N,S,dtype=torch.long); tl=torch.zeros(N,dtype=torch.long)
for i,c in enumerate(ci): tt_[i,:len(c)]=torch.tensor(c); tl[i]=len(c)
lp=emis.unsqueeze(1).expand(T,N,Vsz)
il=torch.full((N,),T,dtype=torch.long)
loss=F.ctc_loss(lp,tt_,il,tl,blank=0,reduction='none',zero_infinity=True).numpy()
print(f"═══ 对 {N} 个【文本】候选算 CTC 分数 ═══")
print("  CTC loss = -log P(这串字母 | 这段音频)，越小越像\n")
o=np.argsort(loss)
print("  排名 | 候选词    CTC loss | 转成概率")
sm=np.exp(-(loss-loss.min())); sm/=sm.sum()
for r,i in enumerate(o[:10],1):
    mk=" ← 目标词" if cw[i]==WORD else ""
    print(f"  {r:>3}  | {cw[i]:<9} {loss[i]:>8.3f} | {sm[i]:.4f}{mk}")
rk=int(np.where(o==cw.index(WORD))[0][0])+1 if WORD in cw else -1
print(f"\n  目标词 {WORD} 排名第 {rk} / {N}")
cb=np.array([base_of(w) for w in cw]); ct=np.array([tone_of(w) for w in cw])
Pword=float(sm[cb==tb].sum())
tsel=cb==tb
s2=np.exp(-(loss[tsel]-loss[tsel].min())); s2/=s2.sum()
Ptone=float(s2[ct[tsel]==tt].sum())
print(f"\n═══ 两个分数 ═══")
print(f"  P_word = {Pword:.4f}   (目标词根 {tb} 在全部候选里的概率和)")
print(f"  P_tone = {Ptone:.4f}   (目标声调 '{tt or '无'}' 在 {tsel.sum()} 个同词根变体里的概率和)")
