#!/usr/bin/env python3
"""部署用 F0 声调后验计算 —— 与 CTC 的 P_tone 融合，补声调判别短板。
用法(库): from f0_infer import F0ToneScorer; s=F0ToneScorer('f0_tone_clf.npz')
          p = s.posterior(wav16k_float32, target_tone_letter)"""
import numpy as np
TONE={"b":1,"":2,"s":3,"j":4,"v":5,"g":6,"m":7}
VALID=set("bsjvgm"); NPTS=16

def tone_id(word):
    w=(word or "").strip().lower()
    return TONE.get(w[-1],2) if w and w[-1] in VALID else 2

def _contour_feats(logf0_norm):
    n=len(logf0_norm); x=np.linspace(0,1,n); xs=np.linspace(0,1,NPTS)
    resamp=np.interp(xs,x,logf0_norm)
    c=np.polyfit(x,logf0_norm,2)
    return np.concatenate([resamp,[float(logf0_norm.mean()),float(logf0_norm.std()),
        float(logf0_norm[0]),float(logf0_norm[-1]),
        float(logf0_norm.max()-logf0_norm.min()),float(c[1]),float(c[0])]]).astype(np.float32)

class F0ToneScorer:
    def __init__(self, clf_path='f0_tone_clf.npz'):
        d=np.load(clf_path,allow_pickle=True)
        self.W=d['W']; self.b=d['b']; self.mu=d['mu']; self.sd=d['sd']
        self._spk_stat={}          # 会话内说话人归一化统计（在线累积）

    def raw_logf0(self, wav, sr=16000):
        import librosa
        f0,vf,vp=librosa.pyin(np.asarray(wav,dtype=np.float32),fmin=70,fmax=400,sr=sr,frame_length=512)
        voiced=~np.isnan(f0)
        vr=float(voiced.mean()) if len(f0) else 0.0
        mvp=float(np.nanmean(vp)) if vp is not None and len(vp) else 0.0
        f0v=f0[voiced]
        return (np.log(f0v).astype(np.float32) if len(f0v)>=3 else None), vr, mvp

    def fit_speaker(self, logf0_list, speaker="_"):
        """用该说话人本场的若干条录音估计基准音高（>=10 条即可稳定）"""
        vals=[x for x in logf0_list if x is not None and len(x)]
        if not vals: return False
        a=np.concatenate(vals); self._spk_stat[speaker]=(float(a.mean()),float(a.std()+1e-6))
        return True

    def posterior(self, wav, word, speaker="_", sr=16000):
        """返回 (目标声调后验, 全部7类后验)。
        注意: 训练特征是"说话人级"归一化 —— 部署时应先对本场全部录音调用 fit_speaker，
        否则退化为单条自身归一化（丢失平调的音高高低信息）。"""
        return self.posterior_from_logf0(*self.raw_logf0(wav,sr),word,speaker)

    def posterior_from_logf0(self, lf, vr, mvp, word, speaker="_"):
        """同 posterior，但直接用 raw_logf0 的结果（会话打分时每条只需跑一次 pYIN）"""
        if lf is None:
            return 0.0, np.full(7,1/7,dtype=np.float32)
        mu,sd=self._spk_stat.get(speaker,(float(lf.mean()),float(lf.std()+1e-6)))
        feats=np.concatenate([_contour_feats((lf-mu)/sd),[vr,mvp]]).astype(np.float32)
        x=(feats-self.mu)/self.sd
        z=x@self.W+self.b; z=z-z.max(); p=np.exp(z); p/=p.sum()
        return float(p[tone_id(word)-1]), p
