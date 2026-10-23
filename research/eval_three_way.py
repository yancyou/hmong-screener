#!/usr/bin/env python3
"""三路融合评估: CTC(字母) + F0(音高) + VQ(发声态)
重点验证 VQ 能否救回 j-g / g / m 这些发声态相关的失效点。"""
import numpy as np,json,csv,glob,collections
from scipy.stats import rankdata

def auc(sc,y):
    p=int((y==1).sum()); n=int((y==0).sum())
    if p<3 or n<3: return float('nan')
    r=rankdata(sc); return 1-(r[y==1].sum()-p*(p+1)/2)/(p*n)

# --- 载入三路 ---
f0=np.load('f0_partial.npz',allow_pickle=True) if glob.glob('f0_full32.npz')==[] else np.load('f0_full32.npz',allow_pickle=True)
wav_f0=f0['wav']; P_f0=np.load('P_f0_partial.npy') if glob.glob('P_f0_full.npy')==[] else np.load('P_f0_full.npy')
lab_f0=f0['label']; spk_f0=f0['speaker']; yt_f0=f0['y_tone']; X_f0=f0['X']
ctc=json.load(open('ctc_by_wav.json'))
VQ={}
for p in sorted(glob.glob('vq_shard*.npz')):
    d=np.load(p,allow_pickle=True)
    for w,x in zip(d['wav'],d['X']): VQ['full32_16k/'+w if not w.startswith('full32_16k/') else w]=x
print(f"CTC {len(ctc)}  F0 {len(wav_f0)}  VQ {len(VQ)}")
if not VQ: raise SystemExit("VQ 分片尚未就绪")

sel=np.array([(w in ctc) and (w in VQ) for w in wav_f0])
print(f"三路齐全: {sel.sum()}")
wav=wav_f0[sel]; lab=lab_f0[sel]; spk=spk_f0[sel]; yt=yt_f0[sel]
Pt=np.array([ctc[w][1] for w in wav]); Pw=np.array([ctc[w][0] for w in wav])
Pf=P_f0[sel]
Xv=np.stack([VQ[w] for w in wav])
Xf=X_f0[sel]

# --- 训 VQ+F0 联合声调分类器（四折，仅用正确样本）---
folds=json.load(open('folds.json'))
fo={s:k for k in range(4) for s in folds[f"fold{k}"]["test"]}
fa=np.array([fo.get(s,-1) for s in spk])
def train_sm(Xt,y,K,ep=500,lr=0.5,l2=3e-4):
    W=np.zeros((Xt.shape[1],K)); b=np.zeros(K); Y=np.eye(K)[y]
    for _ in range(ep):
        z=Xt@W+b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        g=(p-Y)/len(y); W-=lr*(Xt.T@g+l2*W); b-=lr*g.sum(0)
    return W,b
for nm,X in [("VQ单独",Xv),("F0+VQ",np.hstack([Xf,Xv]))]:
    mu,sd=X.mean(0),X.std(0)+1e-9; Xn=(X-mu)/sd
    P=np.zeros(len(X))
    for te in range(4):
        tr=(fa!=te)&(fa>=0)&(lab==1); m=fa==te
        if tr.sum()<100 or m.sum()==0: continue
        W,b=train_sm(Xn[tr],yt[tr]-1,7)
        z=Xn[m]@W+b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        P[m]=p[np.arange(m.sum()),yt[m]-1]
    np.save(f'P_{nm.replace("+","_")}.npy',P)
    s2=(lab==1)|(lab==2); y2=(lab[s2]==2).astype(int)
    print(f"\n{nm} 声调错 AUC = {auc(P[s2],y2):.4f}")
    globals()['P_'+nm.replace('+','_')]=P

# --- 标注可靠性（排除单源孤证，避免噪声淹没信号）---
srcmap={}
for r in csv.DictReader(open('full32_merged.csv')):
    srcmap['full32_16k/'+r['wav_file']]=r.get('src','')
SRC=np.array([srcmap.get(w,'') for w in wav])
RELIABLE=(lab==1)|((lab==2)&(SRC!='union-err'))
print(f"\n可靠标注: 声调错 {int(((lab==2)&(SRC!='union-err')).sum())} 条 "
      f"(排除单源孤证 {int(((lab==2)&(SRC=='union-err')).sum())} 条)")

# --- 融合对比（仅用可靠标注）---
s2=RELIABLE&((lab==1)|(lab==2)); y2=(lab[s2]==2).astype(int)
rc=rankdata(Pt[s2])/s2.sum(); rf=rankdata(Pf[s2])/s2.sum()
rv=rankdata(globals()['P_F0_VQ'][s2])/s2.sum()
print(f"\n{'方案':<22}{'AUC':>9}")
print("-"*32)
print(f"{'CTC 单独':<22}{auc(Pt[s2],y2):>9.4f}")
print(f"{'CTC+F0 (0.4)':<22}{auc(0.6*rc+0.4*rf,y2):>9.4f}")
best=(0,None)
for w2 in (0.2,0.3,0.4,0.5):
    a=auc((1-w2)*rc+w2*rv,y2)
    print(f"{'CTC+(F0,VQ) '+str(w2):<22}{a:>9.4f}")
    if a>best[0]: best=(a,w2)
print(f"\n最优: 权重 {best[1]}  AUC {best[0]:.4f}")

# --- 按混淆对（重点看 j-g）---
note={}
for r in csv.DictReader(open('full32_index.csv')):
    note['full32_16k/'+r['wav_file']]=r.get('note','')
fused=(1-best[1])*rc+best[1]*rv
idx=np.where(s2)[0]
pos=(lab[s2]==2)
o=np.argsort(fused); ss=fused[o]; pp=pos[o]
P_=int(pp.sum()); cp=np.cumsum(pp); cut=np.where(np.r_[ss[1:]!=ss[:-1],True])[0]
j=np.where((P_-cp[cut])/P_<=0.10)[0][0]; THR=ss[cut[j]]
byp=collections.defaultdict(lambda:[0,0]); byp0=collections.defaultdict(lambda:[0,0])
rc_thr=None
o0=np.argsort(rc); s0=rc[o0]; p0=pos[o0]; cp0=np.cumsum(p0)
cut0=np.where(np.r_[s0[1:]!=s0[:-1],True])[0]
j0=np.where((P_-cp0[cut0])/P_<=0.10)[0][0]; THR0=s0[cut0[j0]]
for k,i in enumerate(idx):
    if not pos[k]: continue
    n=note.get(wav[i],'').strip()
    if not n: continue
    byp[n][1]+=1; byp0[n][1]+=1
    if fused[k]<THR: byp[n][0]+=1
    if rc[k]<THR0: byp0[n][0]+=1
# 按说话人（可靠标注）
print(f"\n{'说话人':<10}{'条数':>6}{'CTC':>8}{'F0':>8}{'三路':>8}   变化")
print("-"*48)
prs=[]
for sk in sorted(set(spk[s2])):
    mm=s2&(spk==sk); yy=(lab[mm]==2).astype(int)
    if yy.sum()<5: continue
    r1=rankdata(Pt[mm])/mm.sum(); r2=rankdata(Pf[mm])/mm.sum()
    r3=rankdata(globals()['P_F0_VQ'][mm])/mm.sum()
    a1=auc(Pt[mm],yy); a2=auc(Pf[mm],yy); a3=auc((1-best[1])*r1+best[1]*r3,yy)
    prs.append((a1,sk,int(yy.sum()),a2,a3))
prs.sort()
for a1,sk,n,a2,a3 in prs:
    d=a3-a1; mk="⬆⬆" if d>0.1 else ("⬆" if d>0.02 else ("⬇" if d<-0.02 else ""))
    print(f"{sk:<10}{n:>6}{a1:>8.3f}{a2:>8.3f}{a3:>8.3f}   {d:+.3f} {mk}")
if prs:
    print(f"\n最差: CTC {min(p[0] for p in prs):.3f} -> 三路 {min(p[4] for p in prs):.3f}")
    print(f"中位: CTC {np.median([p[0] for p in prs]):.3f} -> 三路 {np.median([p[4] for p in prs]):.3f}")

print(f"\n{'混淆对':<9}{'CTC':>9}{'三路':>9}{'变化':>7}")
print("-"*36)
for k,(c,n) in sorted(byp.items(),key=lambda x:-x[1][1])[:10]:
    c0=byp0[k][0]
    mk=" ⬆" if c>c0 else (" ⬇" if c<c0 else "")
    print(f"{k:<9}{c0:>4}/{n:<4}{c:>4}/{n:<4}{c-c0:>+5}{mk}")
