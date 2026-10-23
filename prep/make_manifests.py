#!/usr/bin/env python3
"""full32 v3: 由 full32_index.csv + folds.json 生成 fairseq manifest + 评估标签.
   每 fold: test=8人(不进manifest), valid=2人(训侧最少错误者,label-1), train=其余22人
   变体: rlz = label1 + realized(label2/3, suspect=0) | l1 = 仅 label1
   final: 无 test, valid=2人, train=30人 (部署用模型)
输出: manifest_full32/{foldK_rlz,foldK_l1,final_rlz}/ + labels/ + 共享 dict.ltr.txt"""
import csv,json,os,collections,argparse
ap=argparse.ArgumentParser()
ap.add_argument("--index",default="data/full32_index.csv",help="v1 (deployed): data/full32_merged.csv (pre-merge `label`); v2: data/full32_index.csv; 33 speakers: data/full33_index.csv")
ap.add_argument("--folds",default="data/folds.json")
ap.add_argument("--out",default="build/manifest_full32")
ap.add_argument("--dict",default=None,help="reuse this dict.ltr.txt instead of counting letters; the 33-speaker runs reused the 32-speaker dict (deploy/assets/dict.ltr.txt) so every model shares one output-layer order")
a=ap.parse_args()

rows=list(csv.DictReader(open(a.index)))
folds=json.load(open(a.folds))
for r in rows:
    r['label']=int(r['label']); r['suspect']=int(r['suspect']); r['frames']=int(r['frames'])
spk_err=collections.Counter()
for r in rows:
    if r['label'] in (2,3): spk_err[r['speaker']]+=1
all_spk=sorted({r['speaker'] for r in rows})

def transcript(r,variant):
    if r['label']==1: return r['word']
    if variant=='rlz' and r['label'] in (2,3) and not r['suspect'] and r['realized']:
        return r['realized']
    return None

def write_split(d,name,items):
    with open(f"{d}/{name}.tsv",'w') as ft, open(f"{d}/{name}.ltr",'w') as fl, open(f"{d}/{name}.wrd",'w') as fw:
        ft.write("__DATA_ROOT__\n")
        for r,txt in items:
            ft.write(f"{r['wav_file']}\t{r['frames']}\n")
            fl.write(" ".join(txt)+" |\n"); fw.write(txt+"\n")

# 共享字典（覆盖 word+realized 全部字母）
cnt=collections.Counter(); nw=0
for r in rows:
    if r['label']==1: texts={r['word']}
    elif r['label'] in (2,3) and not r['suspect']: texts={r['word'], r['realized'] or r['word']}
    else: continue                      # label-0/suspect 不进字典
    for t in texts:
        if not t.isalpha(): continue
        for ch in t: cnt[ch]+=1
    nw+=1
os.makedirs(a.out,exist_ok=True)
if a.dict:
    import shutil; shutil.copy(a.dict,f"{a.out}/dict.ltr.txt")
else:
    with open(f"{a.out}/dict.ltr.txt",'w') as f:
        f.write(f"| {nw}\n")
        for ch,c in cnt.most_common(): f.write(f"{ch} {c}\n")

def build(dirname,test_set,variant):
    d=f"{a.out}/{dirname}"; os.makedirs(d,exist_ok=True)
    rest=[s for s in all_spk if s not in test_set]
    valid_spk=set(sorted(rest,key=lambda s:(spk_err[s],s))[:2])
    train_spk=set(rest)-valid_spk
    tr=[];va=[]
    for r in rows:
        if r['speaker'] in test_set: continue
        txt=transcript(r,variant)
        if txt is None: continue
        if r['speaker'] in valid_spk:
            if r['label']==1: va.append((r,txt))
        else: tr.append((r,txt))
    write_split(d,'train',tr); write_split(d,'valid',va)
    os.system(f"cp {a.out}/dict.ltr.txt {d}/")
    with open(f"{d}/split.json",'w') as f:
        json.dump({"test":sorted(test_set),"valid":sorted(valid_spk),
                   "train":sorted(train_spk),"variant":variant,
                   "n_train":len(tr),"n_valid":len(va),
                   "n_train_realized":sum(1 for r,_ in tr if r['label']!=1)},f,indent=1)
    return len(tr),len(va),sorted(valid_spk)

print(f"{'目录':<14}{'train':>7}{'valid':>7}  valid说话人")
for k in range(4):
    test=set(folds[f"fold{k}"]["test"])
    for v in ("rlz","l1"):
        nt,nv,vs=build(f"fold{k}_{v}",test,v)
        print(f"fold{k}_{v:<9}{nt:>7}{nv:>7}  {vs}")
nt,nv,vs=build("final_rlz",set(),"rlz")
print(f"{'final_rlz':<14}{nt:>7}{nv:>7}  {vs}")

# 评估标签（closed_vocab_verify 兼容: wav_file,word,label,speaker,list_n,sheet,word_index + suspect）
os.makedirs(f"{a.out}/labels",exist_ok=True)
for k in range(4):
    test=set(folds[f"fold{k}"]["test"])
    with open(f"{a.out}/labels/fold{k}_heldout.csv",'w',newline='') as f:
        w=csv.writer(f); w.writerow(["wav_file","word","label","speaker","list_n","sheet","word_index","suspect","realized"])
        i=0
        for r in rows:
            if r['speaker'] in test and r['label'] in (1,2,3):
                w.writerow([f"full32_16k/{r['wav_file']}",r['word'],r['label'],r['speaker'],
                            r['list_n'],"auto",i,r['suspect'],r['realized']]); i+=1
with open(f"{a.out}/labels/uncertain.csv",'w',newline='') as f:
    w=csv.writer(f); w.writerow(["wav_file","word","speaker","note"])
    for r in rows:
        if r['label']==0: w.writerow([f"full32_16k/{r['wav_file']}",r['word'],r['speaker'],r['note']])
print("\nlabels/ 已生成")
