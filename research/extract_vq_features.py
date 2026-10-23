#!/usr/bin/env python3
"""发声态(voice quality)特征提取 —— 针对 g(气声)/m(紧喉) 的判别短板。

F0 轮廓只描述音高，但 Hmong 的 g(breathy) 和 m(creaky) 区别在**发声方式**:
  breathy -> 声门漏气, 高频噪声多, 谐波弱, H1-H2 大
  creaky  -> 声带紧闭不规则, 周期性差, CPP 低, jitter 大
提取 5 类共 14 维:
  1) 谱倾斜  H1-H2, H1-A1, H1-A3  (声门源特征)
  2) CPP     倒谱峰突出度 (周期性强度, 气声/紧喉都会降低)
  3) HNR     谐噪比 (气声显著降低)
  4) 周期扰动 jitter / shimmer 近似
  5) 高频能量比 + 谱质心/滚降 (噪声成分)
输出 vq_features.npz: X[N,14], wav[N]
"""
from __future__ import annotations
import argparse,csv,os,sys
# 单线程 BLAS/numba: fork 出来的 16 个子进程各自再开线程池会死锁 (上一轮 7/8 分片卡满 72h)
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import multiprocessing as mp
import numpy as np, soundfile as sf

def _safe(x,d=0.0):
    return float(x) if np.isfinite(x) else d

def vq_feats(path):
    try:
        y,sr=sf.read(path,dtype='float32')
        if y.ndim>1: y=y.mean(-1)
        if len(y)<400: return np.zeros(14,dtype=np.float32)
        y=y-y.mean()
        mx=np.abs(y).max()
        if mx>0: y=y/mx
        import librosa
        f0,vflag,vprob=librosa.pyin(y,fmin=70,fmax=400,sr=sr,frame_length=512)
        voiced=~np.isnan(f0)
        if voiced.sum()<3: return np.zeros(14,dtype=np.float32)
        f0m=float(np.nanmedian(f0))
        # --- 谱特征(仅浊音帧) ---
        S=np.abs(librosa.stft(y,n_fft=1024,hop_length=256))+1e-10
        freqs=librosa.fft_frequencies(sr=sr,n_fft=1024)
        nf=min(S.shape[1],len(voiced))
        Sv=S[:,:nf][:,voiced[:nf]] if voiced[:nf].any() else S[:,:nf]
        if Sv.shape[1]==0: Sv=S
        Smean=Sv.mean(1)
        def band(lo,hi):
            m=(freqs>=lo)&(freqs<hi)
            return _safe(20*np.log10(Smean[m].max())) if m.any() else 0.0
        H1=band(max(f0m*0.7,60), f0m*1.3)          # 基频谐波
        H2=band(f0m*1.7, f0m*2.3)                  # 二次谐波
        A1=band(300,900); A3=band(2000,3500)
        tilt=[H1-H2, H1-A1, H1-A3]
        # --- CPP (倒谱峰突出度) ---
        logS=np.log(Smean)
        ceps=np.fft.irfft(logS)
        q0,q1=int(sr/400/ (sr/1024)),int(sr/70/(sr/1024))
        q0=max(2,min(q0,len(ceps)//2)); q1=max(q0+2,min(q1,len(ceps)//2))
        seg=ceps[q0:q1]
        if len(seg)>3:
            x=np.arange(len(seg)); a,b=np.polyfit(x,seg,1)
            cpp=_safe((seg-(a*x+b)).max())
        else: cpp=0.0
        # --- HNR 近似(自相关) ---
        fl=int(0.03*sr); hop=int(0.015*sr); hn=[]
        for st in range(0,max(1,len(y)-fl),hop):
            fr=y[st:st+fl]
            if len(fr)<fl or np.abs(fr).max()<1e-3: continue
            ac=np.correlate(fr,fr,'full')[fl-1:]
            if ac[0]<=0: continue
            ac=ac/ac[0]
            lo=int(sr/400); hi=min(int(sr/70),len(ac)-1)
            if hi>lo+1:
                p=ac[lo:hi].max()
                p=min(max(p,1e-4),0.9999)
                hn.append(10*np.log10(p/(1-p)))
        hnr=_safe(np.mean(hn)) if hn else 0.0
        # --- jitter / shimmer 近似 ---
        f0v=f0[voiced]
        jit=_safe(np.mean(np.abs(np.diff(f0v)))/max(np.mean(f0v),1e-6)) if len(f0v)>2 else 0.0
        env=np.abs(librosa.stft(y,n_fft=512,hop_length=128)).sum(0)
        shm=_safe(np.mean(np.abs(np.diff(env)))/max(np.mean(env),1e-9)) if len(env)>2 else 0.0
        # --- 高频能量与谱形 ---
        tot=Smean.sum()+1e-10
        hf=_safe(Smean[freqs>=2000].sum()/tot)
        hf4=_safe(Smean[freqs>=4000].sum()/tot)
        cent=_safe(float((freqs*Smean).sum()/tot))
        csum=np.cumsum(Smean); roll=_safe(float(freqs[np.searchsorted(csum,0.85*csum[-1])])) if csum[-1]>0 else 0.0
        vr=float(voiced.mean()); mvp=_safe(np.nanmean(vprob))
        return np.array(tilt+[cpp,hnr,jit,shm,hf,hf4,cent/1000.0,roll/1000.0,vr,mvp,f0m/100.0],dtype=np.float32)
    except Exception:
        return np.zeros(14,dtype=np.float32)

def _work(a): return a[0],vq_feats(a[1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels-csv",required=True); ap.add_argument("--data-dir",default=".")
    ap.add_argument("--out",required=True); ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args()
    rows=[r for r in csv.DictReader(open(a.labels_csv)) if (r.get('label') or '').strip() in ('1','2','3','4')]
    print(f"rows: {len(rows)}",flush=True)
    tasks=[(i,os.path.join(a.data_dir,r['wav_file'])) for i,r in enumerate(rows)]
    X=np.zeros((len(rows),14),dtype=np.float32)
    with mp.get_context("spawn").Pool(a.workers) as p:      # spawn: 不继承父进程的线程/锁状态
        for k,(i,f) in enumerate(p.imap_unordered(_work,tasks,chunksize=32),1):
            X[i]=f
            if k%500==0: print(f"  {k}/{len(rows)}",flush=True)
    np.savez(a.out,X=X,wav=np.array([r['wav_file'] for r in rows]))
    z=(np.abs(X).sum(1)==0).sum()
    print(f"X: {X.shape}, 提取失败 {z}")
if __name__=="__main__": main()
