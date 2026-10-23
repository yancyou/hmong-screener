#!/usr/bin/env python3
"""Phase A: 从下载 index.tsv 解析标注 -> full32_parse.tsv + confusion_priors.json + folds.json
   规则（已在 1,318 条上验证 98.5% 一致）:
     无 note                  -> label 1, realized=word
     单字母对 X-Y (X,Y∈声调)  -> label 2, realized=base+Y('n'表示无调字母)
     词对 A-B (A==word)       -> label 3, realized=B
     含逗号/分号/其余         -> label 0 (存疑, 不进训练; 单独评估层)
     X 与目标声调不符 / A≠word -> 保留 label 但 suspect=1 (不进 realized 训练)
"""
import re,json,collections,os
STAGING=os.environ.get("HMONG_STAGING","/staging/x/xouyang7")
OUT=os.environ.get("HMONG_BUILD","build"); os.makedirs(OUT,exist_ok=True)   # outputs go to build/, never over data/
VALID_TONE=set("bsjvgm"); TONE_OR_N=VALID_TONE|{"n"}
def base_of(w): w=w.strip().lower(); return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w): w=w.strip().lower(); return w[-1] if w and w[-1] in VALID_TONE else "n"

rows=[]
for l in open(f"{STAGING}/hmong_segments/index.tsv"):
    key,orig,size,shard,md5=l.rstrip('\n').split('\t')
    parts=orig.split('/'); spk=parts[0].lower()
    fn=parts[-1]; wl=parts[1] if len(parts)>2 else ''
    stem,ext=(fn.rsplit('.',1)+[''])[:2]; ext=ext.lower()
    if ext!='wav': continue          # 84 个 mp4 跳过
    # --- 文件名规整: 双括号/尾部-2重复项/缺右括号 ---
    stem=re.sub(r'\[\[','[',stem)
    stem=re.sub(r'(\])\s*[-_ ]*\d+\s*$',r'\1',stem)      # 'nug [1]-2' -> 'nug [1]'
    if stem.count('[')>1:                                    # 'nros [1] [1]' -> 词部去括号
        head,_,tail=stem.rpartition('[')
        head=re.sub(r'\s*\[[^\]]*\]?','',head)
        stem=head+'['+tail
    if '[' in stem and not re.search(r'[\]\}]\s*$',stem):   # 缺右括号补上
        stem=stem+']'
    m=re.match(r'^\s*(.+?)\s*\[\s*([0-9]+)\s*(?:[;,]\s*(.*?))?\s*[\]\}]\s*$',stem)
    if not m:
        rows.append(dict(orig=orig,shard=shard,spk=spk,wl=wl,word=stem.strip().lower(),
                         rep=0,note='',label=0,realized='',suspect=1)); continue
    word=m.group(1).strip().lower().replace(' ',''); word=re.sub(r'\d+$','',word)
    rep=int(m.group(2)); note=(m.group(3) or '').strip().lower()
    if not word.isalpha():
        rows.append(dict(orig=orig,shard=shard,spk=spk,wl=wl,word=word,rep=rep,note=note,
                         label=0,realized='',suspect=1,listn=0)); continue
    wln=re.search(r'(\d+)\s*$',wl); listn=int(wln.group(1)) if wln else 0
    lab,realized,suspect=1,word,0
    if note:
        if ',' in note or ';' in note: lab,realized,suspect=0,'',0
        elif re.fullmatch(r'[a-z]-[a-z]',note):
            X,Y=note[0],note[2]
            if X in TONE_OR_N and Y in TONE_OR_N:
                lab=2; realized=base_of(word)+('' if Y=='n' else Y)
                if X!=tone_of(word): suspect=1
            else: lab,realized,suspect=0,'',0
        elif re.fullmatch(r'[a-z]{2,}-[a-z]{2,}',note):
            A,B=note.split('-'); lab=3; realized=B
            if A!=word and base_of(A)!=base_of(word): suspect=1
        else: lab,realized,suspect=0,'',0
    rows.append(dict(orig=orig,shard=shard,spk=spk,wl=wl,word=word,rep=rep,note=note,
                     label=lab,realized=realized,suspect=suspect,listn=listn))

with open(f'{OUT}/full32_parse.tsv','w') as f:
    f.write("orig\tshard\tspeaker\tlist_n\tword\trep\tnote\tlabel\trealized\tsuspect\n")
    for r in rows:
        f.write(f"{r['orig']}\t{r['shard']}\t{r['spk']}\t{r.get('listn',0)}\t{r['word']}\t{r['rep']}\t{r['note']}\t{r['label']}\t{r['realized']}\t{r['suspect']}\n")

C=collections.Counter((r['label'],r['suspect']) for r in rows)
print("解析结果 (label,suspect):",dict(C))
print(f"总计 {len(rows)} wav;  可训练 realized(label2/3, suspect=0): "
      f"{sum(1 for r in rows if r['label'] in (2,3) and not r['suspect'])}")

# --- 混淆先验 ---
tone_pairs=collections.Counter(); word_pairs=collections.Counter()
for r in rows:
    if r['suspect']: continue
    if r['label']==2: tone_pairs[f"{tone_of(r['word'])}>{tone_of(r['realized']) if r['realized'] and r['realized'][-1] in VALID_TONE else 'n'}"]+=1
    elif r['label']==3: word_pairs[f"{r['word']}>{r['realized']}"]+=1
onset_sub=collections.Counter()
for k,c in word_pairs.items():
    a,b=k.split('>')
    ba,bb=base_of(a),base_of(b)
    # 送气/onset 替换模式
    if ba.replace('h','')==bb.replace('h',''): onset_sub['h-toggle']+=c
    else: onset_sub['other']+=c
json.dump({"tone_pairs":dict(tone_pairs.most_common()),
           "word_pairs":dict(word_pairs.most_common()),
           "onset_pattern":dict(onset_sub)},
          open(f'{OUT}/confusion_priors.json','w'),indent=1,ensure_ascii=False)
print("\n声调混淆 top10:",tone_pairs.most_common(10))
print("onset 模式:",dict(onset_sub))

# --- 4-fold 说话人划分 (两位说话人钉在 fold0 以对齐 June 评估; 名单在私有文件 data/speakers.json, 见 README) ---
per=collections.defaultdict(lambda:[0,0,0])   # clips, tone_err, word_err
for r in rows:
    p=per[r['spk']]; p[0]+=1
    if r['label']==2: p[1]+=1
    elif r['label']==3: p[2]+=1
spks=sorted(per)
folds=[[] for _ in range(4)]; load=[[0,0,0] for _ in range(4)]
pinned={s:int(f) for s,f in json.load(open(os.environ.get('HMONG_SPEAKERS','data/speakers.json')))['fold_pins'].items()}
for s,f in pinned.items():
    folds[f].append(s); load[f]=[load[f][i]+per[s][i] for i in range(3)]
rest=sorted((s for s in spks if s not in pinned),key=lambda s:-(per[s][1]+per[s][2]))
for s in rest:
    cand=sorted(range(4),key=lambda f:(len(folds[f]),load[f][1]+load[f][2],load[f][0]))
    f=cand[0]; folds[f].append(s); load[f]=[load[f][i]+per[s][i] for i in range(3)]
out={}
for f in range(4):
    out[f"fold{f}"]={"test":sorted(folds[f]),
                     "stats":{"clips":load[f][0],"tone_err":load[f][1],"word_err":load[f][2]}}
json.dump(out,open(f'{OUT}/folds.json','w'),indent=1)
print("\n4-fold 划分:")
for f in range(4):
    o=out[f'fold{f}']; print(f"  fold{f}: {o['stats']}  {o['test']}")
