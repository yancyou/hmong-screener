#!/usr/bin/env python3
"""按折构建混淆先验（只用该折训练侧说话人 → 无泄漏）。final 用全部 33 人（部署用）。
   输出 priors33/priors_fold{k}.json, priors33/priors_final.json  (格式同 confusion_priors.json)"""
import csv,json,os,collections
VALID_TONE=set("bsjvgm")
def base_of(w): return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w): return w[-1] if w and w[-1] in VALID_TONE else "n"
rows=list(csv.DictReader(open('data/full33_index.csv'))); folds=json.load(open('data/folds33.json'))
os.makedirs('build/priors33',exist_ok=True)   # canonical copies: data/priors33/
def build(spk_ok):
    tp=collections.Counter(); wp=collections.Counter(); on=collections.Counter()
    for r in rows:
        if r['speaker'] not in spk_ok or r['suspect']=='1': continue
        w=r['word']; rl=r['realized']
        if not rl or rl==w: continue                         # 自相矛盾/无读音的不进先验
        if r['label']=='2': tp[f"{tone_of(w)}>{tone_of(rl)}"]+=1
        elif r['label']=='3':
            wp[f"{w}>{rl}"]+=1
            on['h-toggle' if base_of(w).replace('h','')==base_of(rl).replace('h','') else 'other']+=1
    return {"tone_pairs":dict(tp.most_common()),"word_pairs":dict(wp.most_common()),"onset_pattern":dict(on)}
allspk={r['speaker'] for r in rows}
for k in range(4):
    tr=allspk-set(folds[f'fold{k}']['test']); p=build(tr)
    json.dump(p,open(f'build/priors33/priors_fold{k}.json','w'),indent=1)
    print(f"fold{k}: 训练侧 {len(tr)} 人 -> word_pairs {len(p['word_pairs'])}, tone_pairs {len(p['tone_pairs'])}")
p=build(allspk); json.dump(p,open('build/priors33/priors_final.json','w'),indent=1)
print(f"final: 33 人 -> word_pairs {len(p['word_pairs'])}")
