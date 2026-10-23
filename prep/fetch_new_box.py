#!/usr/bin/env python3
"""把 Box 上新增的 wav（相对 /staging/x/xouyang7/manifest.tsv）下载成一个新分片。
   输出: hmong_segments/shard-00021.tar + index_20260920.tsv (与 index.tsv 同格式)
   用法: BOX_SHARED_NAME=<共享链接 ID> python3 fetch_new_box.py <box_now.json> <cookie_jar>
   共享链接 ID 是数据所在 Box 文件夹链接的最后一段, 不写进代码。"""
import os,re,io,sys,json,time,random,tarfile,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor
ID=os.environ["BOX_SHARED_NAME"]
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DL=f"https://uwmadison.app.box.com/index.php?rm=box_download_shared_file&shared_name={ID}&file_id=f_"
SEG="/staging/x/xouyang7/hmong_segments"; SHARD="shard-00021.tar"
now=json.load(open(sys.argv[1])); COOK=os.path.abspath(sys.argv[2])
old={l.split('\t')[0] for l in open('/staging/x/xouyang7/manifest.tsv')}
todo=sorted([(k,v['path'],v['size']) for k,v in now.items() if k not in old and v['path'].lower().endswith('.wav')],key=lambda t:t[1])
print(f"待下载 {len(todo)} 个 wav, {sum(t[2] for t in todo)/2**20:.1f} MB",flush=True)
def split_ext(fn):
    i=fn.rfind('.'); return (fn[:i],fn[i+1:].lower()) if 0<i<len(fn)-1 else (fn,"bin")
def key_of(path):
    stem,_=split_ext(path)
    p=re.sub(r'[^A-Za-z0-9_/\-]+','_',stem.replace('.','_'))
    return re.sub(r'_+','_',p).strip('_')
def fetch(t,tries=6):
    fid,p,sz=t; why=""
    for k in range(tries):
        r=subprocess.run(["curl","-sL","-A",UA,"-b",COOK,"--max-time","300",DL+fid],capture_output=True)
        b=r.stdout
        if not b: why=f"空响应 rc={r.returncode}"
        elif b[:5]==b'<!DOC': why="返回HTML"
        elif len(b)!=sz: why=f"大小不符 {len(b)}!={sz}"
        elif b[:4]!=b'RIFF': why="非RIFF"
        else: return t,b,None
        time.sleep(min(1.5**k,20)+random.random())
    return t,None,why
tmp=f"{SEG}/{SHARD}.part"; ok=0; fail=[]; keys=set(); rows=[]
with tarfile.open(tmp,'w') as tf, ThreadPoolExecutor(16) as ex:
    for (fid,p,sz),b,why in ex.map(fetch,todo):
        if b is None: fail.append((p,why)); continue
        k=key_of(p)
        assert k not in keys, k; keys.add(k)
        ti=tarfile.TarInfo(k+".wav"); ti.size=len(b); ti.mtime=0; ti.mode=0o644
        tf.addfile(ti,io.BytesIO(b)); ok+=1
        rows.append((k,p,sz,SHARD,hashlib.md5(b).hexdigest()))
        if ok%500==0: print(f"  {ok}/{len(todo)}",flush=True)
if fail:
    print(f"失败 {len(fail)} 个，分片不落地:",fail[:10]); os.remove(tmp); sys.exit(1)
os.replace(tmp,f"{SEG}/{SHARD}")
with open(f"{SEG}/index_20260920.tsv","w") as f:
    for r in rows: f.write("\t".join(map(str,r))+"\n")
# 当前 Box 全量清单（供下次 diff）
with open("/staging/x/xouyang7/manifest_20260920.tsv","w") as f:
    for k,v in sorted(now.items(),key=lambda kv:kv[1]['path']): f.write(f"{k}\t{v['path']}\t{v['size']}\n")
print(f"完成: {ok} 个 wav -> {SEG}/{SHARD} ({os.path.getsize(f'{SEG}/{SHARD}')/2**20:.1f} MB), 失败 0")
