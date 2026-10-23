#!/usr/bin/env python3
"""用网站 server 自己的函数给录音算余弦相似度用的嵌入 (读音频/切静音/归一化/池化与网站完全一致)。
   同时取最终层(-1, 网站默认)和第 20 层(README 里实测用的层)。"""
import sys,csv,time,argparse,numpy as np,torch
ap=argparse.ArgumentParser(); ap.add_argument("--clips",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
sys.path.insert(0,'.')
import api_similarity_server as S
t0=time.time(); S._load_model()
print(f"model loaded {time.time()-t0:.0f}s | device={S.DEVICE} normalize={S.NORMALIZE} trim={S.TRIM_SILENCE} layers={S._num_transformer_layers()}",flush=True)
rows=list(csv.DictReader(open(a.clips)))
Ef=None; E20=None; bad=0; t0=time.time()
for i,r in enumerate(rows):
    try:
        wav,_=S._read_wav_from_path(r['wav_file'])
        ef=S._extract_embedding(wav,layers=None); e20=S._extract_embedding(wav,layers=[20])
    except Exception as e:
        bad+=1; ef=e20=None
        if bad<=3: print("ERR",r['wav_file'],e,flush=True)
    if Ef is None and ef is not None:
        Ef=np.zeros((len(rows),len(ef)),np.float32); E20=np.zeros((len(rows),len(e20)),np.float32)
    if ef is not None: Ef[i]=ef; E20[i]=e20
    if (i+1)%2000==0: print(f"  {i+1}/{len(rows)}  {(i+1)/(time.time()-t0):.1f} clip/s",flush=True)
np.savez(a.out,wav=np.array([r['wav_file'] for r in rows]),E_final=Ef,E_l20=E20)
print(f"saved {a.out} | 失败 {bad} | {time.time()-t0:.0f}s",flush=True)
