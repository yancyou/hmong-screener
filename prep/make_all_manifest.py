#!/usr/bin/env python3
"""all_rlz: 33 人全部进训练（部署用）。valid 仍放两位说话人 (私有文件 data/speakers.json) 的 label-1（⊂ train，只为让 fairseq 跑起来，
   不用它选 checkpoint —— 用 last / 最后 5 个平均）。转写规则与 make_manifests_full32.py 相同。"""
import csv,json,os,shutil
rows=list(csv.DictReader(open('data/full33_index.csv')))
VALID=json.load(open('data/speakers.json'))['all_rlz_valid']
def transcript(r):
    if r['label']=='1': return r['word']
    if r['label'] in ('2','3') and r['suspect']=='0' and r['realized']: return r['realized']
    return None
d='build/manifest_full33/all_rlz'; os.makedirs(d,exist_ok=True)
tr=[(r,t) for r in rows for t in [transcript(r)] if t]
va=[(r,t) for r,t in tr if r['speaker'] in VALID and r['label']=='1']
for name,items in (('train',tr),('valid',va)):
    with open(f"{d}/{name}.tsv",'w') as ft, open(f"{d}/{name}.ltr",'w') as fl, open(f"{d}/{name}.wrd",'w') as fw:
        ft.write("__DATA_ROOT__\n")
        for r,t in items:
            ft.write(f"{r['wav_file']}\t{r['frames']}\n"); fl.write(" ".join(t)+" |\n"); fw.write(t+"\n")
shutil.copy('build/manifest_full33/final_rlz/dict.ltr.txt',f"{d}/dict.ltr.txt")
json.dump({"test":[],"valid":VALID,"valid_in_train":True,"train":sorted({r['speaker'] for r,_ in tr}),
           "variant":"rlz","n_train":len(tr),"n_valid":len(va),
           "n_train_realized":sum(1 for r,_ in tr if r['label']!='1')},open(f"{d}/split.json",'w'),indent=1)
print(f"all_rlz: train {len(tr)} 条 / {len({r['speaker'] for r,_ in tr})} 人, valid {len(va)} (包含在 train 内)")
