#!/usr/bin/env python3
"""我们的 v1 (闭集 CTC) vs 队友的余弦相似度模型 —— 同一批 28 个两边都没听过的说话人, 同一批录音。"""
import csv,json,collections,numpy as np
from scipy.stats import rankdata
C=list(csv.DictReader(open('cmp_clips.csv'))); idx={r['wav_file']:i for i,r in enumerate(C)}
z=np.load('cosine_emb.npz'); assert list(z['wav'])==[r['wav_file'] for r in C]
src={f"full32_16k/{r['wav_file']}":r['src'] for r in csv.DictReader(open('full33_index.csv'))}
o=np.load('eval_combos_scores.npz',allow_pickle=False); oi={w:i for i,w in enumerate(o['wav'])}
refs=collections.defaultdict(list)
for i,r in enumerate(C):
    if 'ref' in r['role']: refs[r['word']].append(i)
def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,bool); p=y.sum(); n=(~y).sum()
    if p<5 or n<5: return np.nan
    r=rankdata(-s); return (r[y].sum()-p*(p+1)/2)/(p*n)
rng=np.random.default_rng(0); words=sorted(refs)
res={}
for L in ("E_final","E_l20"):
    E=z[L]; T=[]
    for i,r in enumerate(C):
        if 'test' not in r['role']: continue
        R=[j for j in refs[r['word']] if C[j]['speaker']!=r['speaker']]
        if len(R)<2: continue
        proto=E[R].mean(0); proto/=np.linalg.norm(proto)+1e-12
        cos=float(E[i]@proto/(np.linalg.norm(E[i])+1e-12))
        # 简单任务对照: 同一条录音 vs 另一个随机词的模板
        while True:                                   # 他词也必须有 ≥2 条来自其他说话人的参考
            w2=words[rng.integers(len(words))]
            R2=[j for j in refs[w2] if C[j]['speaker']!=r['speaker']]
            if w2!=r['word'] and len(R2)>=2: break
        p2=E[R2].mean(0); p2/=np.linalg.norm(p2)+1e-12
        k=oi[r['wav_file']]
        T.append(dict(spk=r['speaker'],lab=int(r['label']),cos=cos,cos_other=float(E[i]@p2/(np.linalg.norm(E[i])+1e-12)),
                      Pw=float(o['v1__Pw'][k]),Pt=float(o['v1__Pt'][k]),U=src[r['wav_file']]=='union-err'))
    res[L]=T
def table(T,tag):
    lab=np.array([t['lab'] for t in T]); spk=np.array([t['spk'] for t in T]); U=np.array([t['U'] for t in T])
    cos=np.array([t['cos'] for t in T]); co=np.array([t['cos_other'] for t in T])
    Pw=np.array([t['Pw'] for t in T]); Pt=np.array([t['Pt'] for t in T]); Pj=Pw*Pt; ok=lab==1
    out={"n":len(T),"n_word":int((lab==3).sum()),"n_tone":int((lab==2).sum()),"n_ok":int(ok.sum())}
    # 简单任务: 正确录音, 本词模板 vs 他词模板 (分数越高越像本词)
    s=np.r_[cos[ok],co[ok]]; y=np.r_[np.ones(ok.sum(),bool),np.zeros(ok.sum(),bool)]
    out["easy_same_vs_other_word_AUC"]=float(1-auc(s,y))     # auc() 以"分数低=阳性"计, 这里取反
    out["cos_mean_same"]=float(cos[ok].mean()); out["cos_mean_other"]=float(co[ok].mean())
    for nm,pl in (("word",[3]),("tone",[2]),("any",[2,3])):
        pos=np.isin(lab,pl); sel=pos|ok
        ours={"word":Pw,"tone":Pt,"any":Pj}[nm]
        out[f"{nm}_ours"]=auc(ours[sel],pos[sel]); out[f"{nm}_cos"]=auc(cos[sel],pos[sel])
        rel=pos&~U; sel2=rel|ok
        out[f"{nm}_ours_rel"]=auc(ours[sel2],rel[sel2]); out[f"{nm}_cos_rel"]=auc(cos[sel2],rel[sel2])
        # 说话人级 bootstrap: 我们 − 余弦
        S=np.array(sorted(set(spk))); G={s_:np.where(spk==s_)[0] for s_ in S}; D=[]
        for _ in range(1000):
            ii=np.concatenate([G[s_] for s_ in rng.choice(S,len(S))]); p_=pos[ii]; sl=p_|ok[ii]
            D.append(auc(ours[ii][sl],p_[sl])-auc(cos[ii][sl],p_[sl]))
        out[f"{nm}_diff_CI"]=[float(np.nanpercentile(D,2.5)),float(np.nanpercentile(D,97.5))]
        # 会话口径: 每个说话人单独算再平均
        po=[auc(ours[(spk==s_)&sel],pos[(spk==s_)&sel]) for s_ in S]; pc=[auc(cos[(spk==s_)&sel],pos[(spk==s_)&sel]) for s_ in S]
        out[f"{nm}_ours_perspk"]=float(np.nanmean(po)); out[f"{nm}_cos_perspk"]=float(np.nanmean(pc))
    return out
R={L:table(res[L],L) for L in res}
json.dump(R,open('cosine_compare.json','w'),indent=1)
for L,r in R.items():
    print(f"\n=== 余弦模型 {'最终层(网站默认)' if L=='E_final' else '第 20 层(README 用的层)'} | 测试 {r['n']} 条: 整词错 {r['n_word']}, 声调错 {r['n_tone']}, 正确 {r['n_ok']} ===")
    print(f"  对照(简单任务: 本词 vs 他词) AUC {r['easy_same_vs_other_word_AUC']:.3f} | 平均余弦 本词 {r['cos_mean_same']:.3f} / 他词 {r['cos_mean_other']:.3f}")
    print(f"  {'任务':<10}{'我们 v1':>9}{'余弦':>8}{'差值95%CI':>22}{'可靠标注 我们/余弦':>20}{'按说话人 我们/余弦':>20}")
    for nm,zh in (("word","整词错"),("tone","声调错"),("any","任意错")):
        ci=r[f'{nm}_diff_CI']
        print(f"  {zh:<10}{r[f'{nm}_ours']:>9.3f}{r[f'{nm}_cos']:>8.3f}{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}{r[f'{nm}_ours_rel']:>12.3f} / {r[f'{nm}_cos_rel']:.3f}{r[f'{nm}_ours_perspk']:>12.3f} / {r[f'{nm}_cos_perspk']:.3f}")
