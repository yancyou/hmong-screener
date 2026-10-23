"""
闭集 CTC 发音评分器 —— 作为 api_similarity_server 的附加模块。

与原有的余弦相似度并存，提供:
  P_word  目标词在其"对手集"中胜出的概率  (整词错检测, AUC 0.918)
  P_tone  目标声调在 7 调中胜出的概率      (声调错检测, AUC 0.880)
  verdict PASS / REVIEW  (阈值: 33 人四折留出交叉验证, 放行内正确率 ≥99.5%)
  数字口径 (2026-09-22): 按折无泄漏先验、部署同款候选全集, 见 FINDINGS.md / select_model.json

不需要参考录音 —— 直接由 CTC 模型对候选词打分。
"""
from __future__ import annotations
import csv, json, os, threading
from typing import Dict, List, Optional
import numpy as np, torch, torch.nn.functional as F

VALID_TONE = list("bsjvgm"); TONES = VALID_TONE + [""]
VOWELS = set("aeiou")

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
    o,r = onset_of(b), rime_of(b)
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

class CTCScorer:
    """惰性初始化；复用 server 已加载的 MODEL/TASK。"""
    def __init__(self, vocab_csv: str, priors_json: str, thresholds_json: str, dict_path: Optional[str] = None):
        self.vocab_csv, self.priors_json, self.th_json = vocab_csv, priors_json, thresholds_json
        self.dict_path = dict_path          # 部署包的 dict.ltr.txt —— 用来核对模型字典
        self._ready = False; self._lock = threading.Lock()

    def _check_dict(self, tgt):
        """模型的字母表必须与训练字典逐位一致, 否则 CTC 索引错位、所有分数都是垃圾
        (典型原因: DATA_DIR 没指向含 dict.ltr.txt 的目录, server 自动合成了 a-z 回退字典)。"""
        if not self.dict_path or not os.path.exists(self.dict_path):
            return
        want = ["<s>", "<pad>", "</s>", "<unk>"] + [l.split()[0] for l in open(self.dict_path) if l.strip()]
        have = [tgt[i] for i in range(len(tgt))]
        if have != want:
            raise RuntimeError(
                f"模型字典与部署包 dict.ltr.txt 不一致 (模型 {len(have)} 个符号, 应为 {len(want)} 个)。"
                f"请把 DATA_DIR(或 CTC_DATA_DIR) 指向含部署包 dict.ltr.txt 的目录, 并确认加载的是 CTC 模型。")

    def _build(self, task):
        self._check_dict(task.target_dictionary)
        vocab=set()
        with open(self.vocab_csv) as f:
            for r in csv.DictReader(f):
                w=(r.get("word") or "").strip().lower()
                if w and w.isalpha(): vocab.add(w)
        self.word_riv={}; self.realized=set()
        if os.path.exists(self.priors_json):
            pri=json.load(open(self.priors_json))
            for k,c in pri.get("word_pairs",{}).items():
                w,rv=k.split(">"); self.word_riv.setdefault(w,set()).add(base_of(rv)); self.realized.add(rv)
        self.thr = 0.81365
        if os.path.exists(self.th_json):
            th=json.load(open(self.th_json))
            self.thr=float(th.get("P_joint",{}).get("pass@0.995",{}).get("thr",self.thr))
        real_bases=sorted({base_of(w) for w in vocab})
        cand={w:True for w in vocab}
        for b in real_bases:
            for t in TONES: cand.setdefault(b+t,False)
            for v in asp_variants(b):
                for t in TONES: cand.setdefault(v+t,False)
        # 先验里实测读成的词也进候选 —— 与离线评估(score_losses.deploy_allowed)的候选全集一致
        for rv in sorted(self.realized):
            if rv.isalpha(): cand.setdefault(rv,False)
        tgt=task.target_dictionary
        cw=[];ci=[]
        for w in cand:
            idx=[tgt.index(c) for c in w]
            if idx and all(i!=tgt.unk() for i in idx): cw.append(w); ci.append(idx)
        self.cw=cw; self.N=len(cw)
        Smax=max(len(c) for c in ci)
        T=torch.zeros(self.N,Smax,dtype=torch.long); L=torch.zeros(self.N,dtype=torch.long)
        for i,c in enumerate(ci): T[i,:len(c)]=torch.tensor(c); L[i]=len(c)
        self.targets=T; self.tlen=L
        self.cb=np.array([base_of(w) for w in cw])
        self.ct=np.array([tone_of(w) for w in cw])
        self.cr=np.array([rime_of(b) for b in self.cb])
        self.ir=np.array([cand[w] for w in cw])
        self.blank=0
        self._ready=True

    def ensure(self, task):
        if not self._ready:
            with self._lock:
                if not self._ready: self._build(task)

    @torch.no_grad()
    def score(self, model, task, wav16k: torch.Tensor, word: str, normalize: bool) -> Dict:
        """wav16k: 1D float tensor @16kHz"""
        self.ensure(task)
        dev=next(model.parameters()).device
        w=(word or "").strip().lower()
        tb,tt=base_of(w),tone_of(w)
        riv={tb}|self.word_riv.get(w,set())|asp_variants(tb)
        msel=np.isin(self.cb,list(riv))|((self.cr==rime_of(tb))&self.ir)
        tsel=(self.cb==tb)
        if not msel.any() and not tsel.any():
            return {"error":f"目标词 {w!r} 不在词表内","P_word":None,"P_tone":None}
        need=np.where(msel|tsel)[0]                       # 只算必要候选(省~97%)
        x=wav16k
        if normalize: x=F.layer_norm(x,x.shape)
        x=x.unsqueeze(0).to(dev)
        pm=torch.zeros(1,x.shape[1],dtype=torch.bool,device=dev)
        o=model(source=x,padding_mask=pm)
        lg=model.get_logits(o) if hasattr(model,"get_logits") else o["encoder_out"]
        if lg.dim()==3: lg=lg[:,0,:] if lg.shape[1]==1 else lg[0]
        emis=F.log_softmax(lg.float(),dim=-1)
        Tn=emis.shape[0]; M=len(need)
        tg=self.targets[need].to(dev); tl=self.tlen[need].to(dev)
        lp=emis.unsqueeze(1).expand(Tn,M,emis.shape[1])
        il=torch.full((M,),Tn,dtype=torch.long,device=dev)
        loss=F.ctc_loss(lp,tg,il,tl,blank=self.blank,reduction='none',zero_infinity=True).cpu().numpy()
        sm=lambda v: np.exp(-(v-v.min()))/np.exp(-(v-v.min())).sum()
        pos={g:k for k,g in enumerate(need)}
        mi=np.array([pos[g] for g in np.where(msel)[0]]) if msel.any() else np.array([],dtype=int)
        ti=np.array([pos[g] for g in np.where(tsel)[0]]) if tsel.any() else np.array([],dtype=int)
        P_word=float(sm(loss[mi])[self.cb[msel]==tb].sum()) if len(mi) else 0.0
        P_tone=float(sm(loss[ti])[self.ct[tsel]==tt].sum()) if len(ti) else 0.0
        P_joint=P_word*P_tone
        return {"P_word":round(P_word,5),"P_tone":round(P_tone,5),
                "P_joint":round(P_joint,6),
                "verdict":"PASS" if P_joint>=self.thr else "REVIEW",
                "n_candidates":int(M),"threshold":self.thr,"error":None}


# ================= F0 声调融合（v2） =================
# 实验结论（32 说话人 / 40,037 条, 会话内秩融合 = 部署口径）:
#   声调错检测 AUC (按说话人平均): CTC 单独 0.821 -> CTC+F0(权重0.4) 0.842;  最差说话人 0.590 -> 0.588
#   单人收益显著: 前三位说话人 +19.2 / +12.1 / +8.1 (第一位 CTC 0.584、F0 0.986, 是典型互补)
F0_FUSION_WEIGHT = 0.4

def fuse_tone(rank_ctc: float, rank_f0: float, w: float = F0_FUSION_WEIGHT) -> float:
    """两路声调分数在会话内的秩空间融合。rank_* ∈ [0,1]，越低越可疑。"""
    return (1.0 - w) * rank_ctc + w * rank_f0
