#!/usr/bin/env python3
"""上线后自检: 把 12 条真实录音发给网站的 /api/ctc_score, 与已验证结果逐条比对。用法: python3 live_check.py https://<app>.fly.dev"""
import sys,os,csv,json,uuid,urllib.request,urllib.error,time,base64
base=sys.argv[1].rstrip('/'); here=os.path.dirname(os.path.abspath(__file__))
# 网站开了登录 (Space secret ACCESS_PASSWORD) 时: 密码放在 ~/.hmong_site_password (权限 600) 或环境变量 SITE_PASSWORD
_pf=os.path.expanduser("~/.hmong_site_password")
_pw=os.environ.get("SITE_PASSWORD") or (open(_pf).read().strip() if os.path.exists(_pf) else "")
AUTH={"Authorization":"Basic "+base64.b64encode(f"{os.environ.get('SITE_USER','hmong')}:{_pw}".encode()).decode()} if _pw else {}
def get(p,t=600):
    try:
        with urllib.request.urlopen(urllib.request.Request(base+p,headers=AUTH),timeout=t) as r: return r.getcode(),r.read().decode()
    except urllib.error.HTTPError as e: return e.code,e.read().decode()[:300]
    except Exception as e: return None,str(e)
for i in range(60):                       # 冷启动: 机器从休眠唤醒 + 加载两个模型
    c,b=get("/healthz",t=120)
    if c==200: break
    time.sleep(10)
print(f"/healthz [{c}] {b[:80]}")
c,b=get("/api/ctc_info"); info=json.loads(b) if c==200 else {}
if c==401: print("✗ 网站要求登录: 把密码写进 ~/.hmong_site_password (chmod 600) 或设 SITE_PASSWORD 后重跑"); sys.exit(1)
print(f"/api/ctc_info [{c}] error={info.get('error')}  validated={'有' if info.get('validated') else '无'}")
rows=list(csv.DictReader(open(f"{here}/clips.csv"))); bnd=uuid.uuid4().hex; parts=[]
for r in rows:
    parts+=[b'--'+bnd.encode()+b'\r\nContent-Disposition: form-data; name="files"; filename="'+os.path.basename(r['wav_file']).encode()+b'"\r\nContent-Type: audio/wav\r\n\r\n'+open(f"{here}/{r['wav_file']}",'rb').read()+b'\r\n',
            b'--'+bnd.encode()+b'\r\nContent-Disposition: form-data; name="words"\r\n\r\n'+r['word'].encode()+b'\r\n']
parts.append(b'--'+bnd.encode()+b'--\r\n')
req=urllib.request.Request(base+"/api/ctc_score",data=b''.join(parts),headers={"Content-Type":f"multipart/form-data; boundary={bnd}",**AUTH})
t0=time.time()
try:
    with urllib.request.urlopen(req,timeout=900) as r: d=json.loads(r.read().decode())
except urllib.error.HTTPError as e:
    print(f"✗ /api/ctc_score 返回 HTTP {e.code}: {e.read().decode()[:200]}"); sys.exit(1)
ref={x['filename']:x for x in json.load(open(f"{here}/api_out_mix12.json"))['results']}
mx=max(max(abs(x['P_word']-ref[x['filename']]['P_word']),abs(x['P_tone']-ref[x['filename']]['P_tone'])) for x in d['results'])
dv=sum(x['verdict']!=ref[x['filename']]['verdict'] for x in d['results'])
print(f"/api/ctc_score: {d['n_scored']} 条, {time.time()-t0:.0f}s, fusion={d.get('fusion')} | 与已验证结果最大差 {mx:.1e}, 判定不同 {dv} 条")
ok = c is not None and info.get('validated') and not info.get('error') and mx<1e-3 and dv==0
print("✓ 网站上线自检通过" if ok else "✗ 自检未通过 —— 见上面各行"); sys.exit(0 if ok else 1)
