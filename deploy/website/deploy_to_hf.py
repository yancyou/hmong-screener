#!/usr/bin/env python3
"""把网站部署到 Hugging Face Spaces 免费档 (cpu-basic: 2 vCPU / 16 GB)。
   私有模型仓库 <user>/hmong-screener-private 存 bundle.tar (代码 + 筛查模型);
   公开 Space <user>/hmong-screener 只有构建配方, 启动时用 Space secret HF_TOKEN 下载 bundle。
   用法: python3 deploy_to_hf.py   (token 读 ~/.hf_token)"""
import os,sys,time,subprocess
from huggingface_hub import HfApi
TOKEN=open(os.path.expanduser("~/.hf_token")).read().strip()
api=HfApi(token=TOKEN)
user=api.whoami()["name"]; print("HF 账号:",user)
priv=f"{user}/hmong-screener-private"; space=f"{user}/hmong-screener"
api.create_repo(priv,repo_type="model",private=True,exist_ok=True)
print("上传 bundle.tar (1.3 GB) 到私有仓库", priv, flush=True)
BUNDLE=os.environ.get("HMONG_BUNDLE","build/hf_bundle.tar")   # made by deploy/website/build_bundle.sh
assert os.path.exists(BUNDLE), f"missing {BUNDLE}: run deploy/website/build_bundle.sh first"
api.upload_file(path_or_fileobj=BUNDLE,path_in_repo="bundle.tar",repo_id=priv,repo_type="model",
                commit_message="code + screening model bundle")
assert api.repo_info(priv,repo_type="model").private, "私有仓库不是 private!"
api.create_repo(space,repo_type="space",space_sdk="docker",private=False,exist_ok=True)
api.add_space_secret(space,"HF_TOKEN",TOKEN)
api.add_space_variable(space,"PRIVATE_REPO",priv)
before=api.repo_info(space,repo_type="space").sha
api.upload_folder(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hf_space"),repo_id=space,repo_type="space",
                  commit_message="Hmong pronunciation screener")
if api.repo_info(space,repo_type="space").sha==before:   # 配方没变就不会重建; bundle 只在启动时下载, 所以手动重启
    print("Space 配方没变, 重启以加载新的 bundle", flush=True); api.restart_space(space)
    for _ in range(12):
        if api.get_space_runtime(space).stage!="RUNNING": break
        time.sleep(10)
print("Space 文件已上传, 等待构建...", flush=True)
t0=time.time(); last=None
while time.time()-t0<3600:
    st=api.get_space_runtime(space).stage
    if st!=last: print(f"  [{time.time()-t0:5.0f}s] {st}", flush=True); last=st
    if st in ("RUNNING","BUILD_ERROR","RUNTIME_ERROR","CONFIG_ERROR","NO_APP_FILE"): break
    time.sleep(20)
host=f"https://{user.lower().replace('_','-').replace('.','-')}-hmong-screener.hf.space"
print("Space 页面: https://huggingface.co/spaces/"+space, "| 网站:", host)
if last!="RUNNING": sys.exit(f"✗ Space 状态 {last} —— 查看构建/运行日志")
r=subprocess.run([sys.executable,os.path.join(os.path.dirname(os.path.abspath(__file__)),"smoke/live_check.py"),host])
sys.exit(r.returncode)
