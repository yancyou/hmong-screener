import numpy as np,csv,glob,json
from scipy.stats import rankdata
lab={}
for r in csv.DictReader(open('data/full32_merged.csv')):
    if r['label_merged'] in ('1','2','3'): lab['full32_16k/'+r['wav_file']]=int(r['label_merged'])
X=[];yt=[];sp=[];src=[]
for p in sorted(glob.glob('build/f0/f0_shard*.npz')):
    d=np.load(p,allow_pickle=True)
    rows=list(csv.DictReader(open(p.replace('.npz','.csv'))))
    if len(rows)!=len(d['X']): continue
    X.append(d['X']); yt.append(d['y_tone']); sp.append(d['speaker']); src+=[r['wav_file'] for r in rows]
X=np.concatenate(X); yt=np.concatenate(yt); sp=np.concatenate(sp); src=np.array(src)
k=np.array([w in lab for w in src]); X,yt,sp,src=X[k],yt[k],sp[k],src[k]
y3=np.array([lab[w] for w in src])
folds=json.load(open('data/folds.json')); fo={s:i for i in range(4) for s in folds[f'fold{i}']['test']}
fa=np.array([fo.get(s,-1) for s in sp])
mu,sd=X.mean(0),X.std(0)+1e-9; Xn=((X-mu)/sd).astype(np.float64)
def tr_sm(Xt,y,K,ep=300,lr=1.0,l2=3e-4):
    W=np.zeros((Xt.shape[1],K)); b=np.zeros(K); Y=np.eye(K)[y]
    for _ in range(ep):
        z=Xt@W+b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        g=(p-Y)/len(y); W-=lr*(Xt.T@g+l2*W); b-=lr*g.sum(0)
    return W,b
P=np.zeros(len(X))
for te in range(4):
    tr=(fa!=te)&(fa>=0)&(y3==1); m=fa==te
    if tr.sum()<100 or m.sum()==0: continue
    W,b=tr_sm(Xn[tr],yt[tr]-1,7)
    z=Xn[m]@W+b; z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True)
    P[m]=q[np.arange(m.sum()),yt[m]-1]
    print(f"  fold{te} done ({m.sum()})",flush=True)
np.save('build/P_f0_full32.npy',P); np.savez('build/f0_full32_meta.npz',sp=sp,y3=y3,src=src,yt=yt)
ctc=json.load(open('data/derived/ctc_by_wav.json'))
Pt=np.array([ctc[w][1] if w in ctc else np.nan for w in src])
have=~np.isnan(Pt); sel=have&((y3==1)|(y3==2)); y=(y3[sel]==2).astype(int)
def auc(s_,yy):
    p=int((yy==1).sum()); n=int((yy==0).sum())
    if p<3 or n<3: return float('nan')
    r=rankdata(s_); return 1-(r[yy==1].sum()-p*(p+1)/2)/(p*n)
rc=rankdata(Pt[sel])/sel.sum(); rf=rankdata(P[sel])/sel.sum()
print(f"\n══ 全32说话人/合并标注 (n={sel.sum()}, 声调错 {y.sum()}) ══")
print(f"  CTC 单独  AUC = {auc(Pt[sel],y):.4f}")
print(f"  F0  单独  AUC = {auc(P[sel],y):.4f}")
spk_list=[s for s in sorted(set(sp[sel])) if ((y3==2)&(sp==s)).sum()>=5]
print(f"\n{'权重':<7}{'整体':>9}{'最差':>8}{'中位':>8}")
for w in (0,0.2,0.3,0.4,0.5):
    per=[]
    for s in spk_list:
        m=sel&(sp==s); yy=(y3[m]==2).astype(int)
        r1=rankdata(Pt[m])/m.sum(); r2=rankdata(P[m])/m.sum()
        per.append(auc((1-w)*r1+w*r2,yy))
    print(f"{w:<7}{auc((1-w)*rc+w*rf,y):>9.4f}{min(per):>8.3f}{np.median(per):>8.3f}")
