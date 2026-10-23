#!/usr/bin/env python3
"""新增分片 -> full33_add_16k.tar (16k mono PCM16, 条目名与 full32_16k.tar 同前缀，可解压到同一目录)
   + full33_index.csv (= full32_index.csv + 新增行)。标签解析规则与 parse_labels.py 逐字一致。"""
import io,os,re,csv,tarfile,collections
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
STAGING=os.environ.get("HMONG_STAGING","/staging/x/xouyang7")
SEG=f"{STAGING}/hmong_segments"; OUT_TAR="build/full33_add_16k.tar"   # canonical copy: $HMONG_STAGING/full33_add_16k.tar
VALID_TONE=set("bsjvgm"); TONE_OR_N=VALID_TONE|{"n"}
def base_of(w): w=w.strip().lower(); return w[:-1] if w and w[-1] in VALID_TONE else w
def tone_of(w): w=w.strip().lower(); return w[-1] if w and w[-1] in VALID_TONE else "n"
def parse(orig):                      # ← 与 parse_labels.py 主循环相同
    parts=orig.split('/'); spk=parts[0].lower(); fn=parts[-1]; wl=parts[1] if len(parts)>2 else ''
    stem,ext=(fn.rsplit('.',1)+[''])[:2]
    stem=re.sub(r'\[\[','[',stem)
    stem=re.sub(r'(\])\s*[-_ ]*\d+\s*$',r'\1',stem)
    if stem.count('[')>1:
        head,_,tail=stem.rpartition('['); head=re.sub(r'\s*\[[^\]]*\]?','',head); stem=head+'['+tail
    if '[' in stem and not re.search(r'[\]\}]\s*$',stem): stem=stem+']'
    wln=re.search(r'(\d+)\s*$',wl); listn=int(wln.group(1)) if wln else 0
    m=re.match(r'^\s*(.+?)\s*\[\s*([0-9]+)\s*(?:[;,]\s*(.*?))?\s*[\]\}]\s*$',stem)
    if not m: return dict(spk=spk,word=stem.strip().lower(),note='',label=0,realized='',suspect=1,listn=listn)
    word=re.sub(r'\d+$','',m.group(1).strip().lower().replace(' ',''))
    note=(m.group(3) or '').strip().lower()
    if not word.isalpha(): return dict(spk=spk,word=word,note=note,label=0,realized='',suspect=1,listn=0)
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
    return dict(spk=spk,word=word,note=note,label=lab,realized=realized,suspect=suspect,listn=listn)
def convert(args):
    name,blob=args
    wav,sr=sf.read(io.BytesIO(blob),dtype='float32')
    if wav.ndim>1: wav=wav.mean(axis=1)
    if sr!=16000: fr=Fraction(16000,sr); wav=resample_poly(wav,fr.numerator,fr.denominator)
    pk=np.abs(wav).max()
    if pk>1.0: wav=wav/pk
    buf=io.BytesIO(); sf.write(buf,wav,16000,subtype='PCM_16',format='WAV')
    return name,buf.getvalue(),len(wav)
def main():
    idx=[l.rstrip('\n').split('\t') for l in open(f"{SEG}/index_20260920.tsv")]
    old=list(csv.DictReader(open('data/full32_index.csv'))); cols=list(old[0].keys())
    start=collections.Counter()
    for r in old:
        i=int(re.match(r'^[^/]+/(\d+)_',r['wav_file']).group(1)); start[r['speaker']]=max(start[r['speaker']],i+1)
    P={orig:parse(orig) for _,orig,_,_,_ in idx}
    per=collections.defaultdict(list)
    for orig,p in P.items(): per[p['spk']].append(orig)
    plan={}
    for spk,os_ in per.items():
        os_.sort(key=lambda o:(P[o]['listn'],o))
        for j,o in enumerate(os_): plan[o]=f"{spk}/{start[spk]+j}_{P[o]['word']}.wav"
    key2orig={k:o for k,o,_,_,_ in idx}
    tf=tarfile.open(f"{SEG}/shard-00021.tar"); batch=[]
    for m in tf:
        k=m.name[:-4]; batch.append((key2orig[k],tf.extractfile(m).read()))
    tf.close()
    rows=[]; os.makedirs('build',exist_ok=True)
    with tarfile.open(OUT_TAR+".part",'w') as out, ProcessPoolExecutor(8) as ex:
        for orig,data,frames in ex.map(convert,batch,chunksize=16):
            rel=plan[orig]; p=P[orig]
            ti=tarfile.TarInfo("full32_16k/"+rel); ti.size=len(data); ti.mtime=0; ti.mode=0o644
            out.addfile(ti,io.BytesIO(data))
            lab=str(p['label']); src='cur' if lab=='0' else 'cur-only'
            rows.append(dict(wav_file=rel,word=p['word'],label=lab,speaker=p['spk'],realized=p['realized'],
                             note=p['note'],suspect=p['suspect'],frames=frames,list_n=p['listn'],
                             orig_path=orig,label_merged=lab,src=src))
    os.replace(OUT_TAR+".part",OUT_TAR)
    allrows=old+rows; allrows.sort(key=lambda r:r['wav_file'])
    assert len({r['wav_file'] for r in allrows})==len(allrows), "wav_file 重名"
    with open('build/full33_index.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(allrows)
    c=collections.Counter((r['speaker'],r['label']) for r in rows)
    print(f"新增 {len(rows)} 条 -> {OUT_TAR} ({os.path.getsize(OUT_TAR)/2**20:.1f} MB)")
    for s in sorted({r['speaker'] for r in rows}):
        print(f"  {s}: " + "  ".join(f"label{l}={c[(s,l)]}" for l in '0123'))
    print(f"full33_index.csv: {len(allrows)} 行, {len({r['speaker'] for r in allrows})} 说话人")
    print("无法解析的文件名:",[r['orig_path'].split('/')[-1] for r in rows if r['label']=='0'])
    fr=[r['frames'] for r in rows]; print(f"frames: min={min(fr)} max={max(fr)} (16k)")
if __name__=="__main__": main()
