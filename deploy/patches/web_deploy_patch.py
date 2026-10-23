#!/usr/bin/env python3
"""把 CTC 打分加进网站镜像 (Dockerfile, 幂等)。在软件根目录运行。
   前提: 部署包已解压到软件根目录, 且模型已放到 ./ctc_checkpoint.pt (即 full32_foldfinal_infer.pt 改名)。
   结果: 网站同时加载两个模型 —— 原余弦模型(CHECKPOINT_PATH, 不变) + CTC 模型(CTC_CHECKPOINT_PATH)。"""
import io,os,sys
P="Dockerfile"; MARK="# ==== CTC closed-vocab scoring (added) ===="
s=io.open(P,encoding="utf-8").read()
if MARK in s: print("Dockerfile 补丁已存在，跳过"); sys.exit(0)
anchor="COPY api_similarity_server.py freeze_layers_hook.py template_manifest_all_sources_with_xlsx.csv ./\n"
if anchor not in s: sys.exit("✗ 没找到 Dockerfile 里复制 server 代码的那一行 —— Dockerfile 结构变了，请手动加入下面这几行")
block=f"""{MARK}
# 模型打进镜像 (不公开上传); 余弦模型照旧由 CHECKPOINT_PATH 提供, 两个模型并存
COPY ctc_scorer.py f0_infer.py f0_tone_clf.npz full33_index.csv priors_final.json deploy_thresholds.json dict.ltr.txt ./
COPY dict.ltr.txt ./ctc_data/dict.ltr.txt
COPY ctc_checkpoint.pt /app/ctc_checkpoint.pt
ENV CTC_CHECKPOINT_PATH=/app/ctc_checkpoint.pt
"""
missing=[f for f in ("ctc_scorer.py","f0_infer.py","f0_tone_clf.npz","full33_index.csv","priors_final.json","deploy_thresholds.json","dict.ltr.txt","ctc_checkpoint.pt") if not os.path.exists(f)]
s=s.replace(anchor,anchor+block,1)
io.open(P,"w",encoding="utf-8").write(s)
print("✓ Dockerfile 已加入 CTC 模型与文件")
if missing: print("⚠ 构建前还需要放好这些文件:", ", ".join(missing))
