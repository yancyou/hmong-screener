# 网站部署（HuggingFace Spaces）

**网址：https://youngxuan-hmong-screener.hf.space**（任何电脑直接打开，不用登录）

| 页面 | 地址 |
|---|---|
| Evaluate（原余弦相似度，首页） | `/` |
| Batch Score（选 **Closed-vocab CTC (+F0)** 就是新筛查模型） | `/#/batch` |
| Collect Data（录音） | `/#/collect` |

Space 管理页：https://huggingface.co/spaces/YoungXuan/hmong-screener （Logs、Settings、重启都在这里）

## 架构

| 部分 | 内容 | 谁能看到 |
|---|---|---|
| 公开 Space `YoungXuan/hmong-screener` | 只有构建配方：`Dockerfile`、`start.sh`、`requirements-deploy.txt`、`README.md`（CHTC 上 `~/hmong/hf_space/`） | 所有人 |
| 私有模型仓库 `YoungXuan/hmong-screener-private` | `bundle.tar`（1.3 GB）：服务器代码、前端、筛查模型、打分文件 | 只有账号主人 |
| Space secret `HF_TOKEN` | 容器启动时用它下载 `bundle.tar`；下载完就从服务器进程的环境里删掉 | 只有账号主人（写入后不可见） |
| Space variable `PRIVATE_REPO` | `YoungXuan/hmong-screener-private` | 账号主人 |

镜像本身不含代码和模型（公开 Space 的镜像任何人都能拉取）。余弦模型 `tianyi0216/hmong-xlsr-checkpoint` 本来就公开，构建时直接下载。
不上传 `full33_index.csv`、`template_manifest*.csv`（含说话人 ID 和路径）；筛查用的词表是只有一列词的 `vocab.csv`。

## 规格与限制

- 免费档 CPU Basic：2 vCPU、16 GB 内存。两个模型同时加载峰值约 4 GB。
- **48 小时没人访问会休眠**；之后任何人打开网址都会自动唤醒，等 1～2 分钟（下载 bundle + 加载两个模型）。
- HuggingFace 现在要求付费账号才能新建 Docker Space；这个 Space 建在 YoungXuan 的 PRO 账号下。
- 速度（线上实测）：每条约 0.53～0.66 秒。194 条一份 103 秒；999 条一次请求 573 秒；1,351 条整份 890 秒（约 15 分钟）。
- 一次上传（一个会话）最多 5,000 条录音（环境变量 `CTC_MAX_FILES` 可调）。
- 同时提交的会话排队、依次处理（2 个 CPU 并行也不会更快；而且 F0 的会话内音高归一化是共享状态，并行会互相覆盖）。
- 打分期间服务器每 10 秒发一个空格保持连接，响应头立即返回，所以 Firefox（默认 5 分钟收不到响应就断开）和网关都不会在长会话中途断开。中途关掉页面，服务器在 10 秒内停止这一份的打分。
- 上传的录音只在内存/临时目录里打分，不保存；容器磁盘每次重启清空。
- 网站是公开的（2026-10-04 决定不设密码）：拿到网址的人都能用。以后要加密码，软件自带登录功能：在 Space Settings → Variables and secrets 加 secret `ACCESS_PASSWORD`（用户名默认 `hmong`，可用 `ACCESS_USER` 改），网页和 API 都要先登录，`/healthz` 除外。开了密码后，部署脚本最后的线上自检要用密码：写进 CHTC 上的 `~/.hmong_site_password`（`chmod 600`）或设环境变量 `SITE_PASSWORD`。

## 更新网站（改了代码或模型之后）

在 CHTC 上（需要一个 **Write** 类型的 HuggingFace token，存在 `~/.hf_token`，权限 600，不要贴到聊天里）：

```bash
cd ~/hmong/web_deploy/hmong_audio_software-main
python3 ctc_endpoint_patch.py && python3 gui_ctc_patch.py && python3 gui_table_patch.py   # 幂等
# 改了前端要重新构建 GUI/dist（node 在 ~/hmong/_swtest/node/bin）
B=/staging/x/xouyang7/hf_bundle.tar
tar --exclude='__pycache__' --exclude='*.pyc' -cf $B api_similarity_server.py freeze_layers_hook.py fairseq GUI/dist \
    ctc_scorer.py f0_infer.py f0_tone_clf.npz priors_final.json deploy_thresholds.json dict.ltr.txt ctc_data vocab.csv ctc_checkpoint.pt
cd ~/hmong && python3 deploy_to_hf.py
```

`deploy_to_hf.py` 会：上传 `bundle.tar` 到私有仓库（只传变化的部分）→ 确认仓库是 private → 更新 Space 配方、secret、variable →
配方没变时主动重启 Space（bundle 只在启动时下载）→ 等到 RUNNING → 用 12 条已知答案的录音做线上自检（`web_deploy/smoke/live_check.py`）。

## 出问题时

- 看日志：Space 管理页 → Logs（服务器把 numba/multipart 的 DEBUG 信息也打出来，量很大；找 `[start]`、`[ctc]`、`Traceback`）。
  启动时的 `FFmpeg extension is not available` 是 torchaudio 探测可选插件，不影响。
- `https://youngxuan-hmong-screener.hf.space/api/ctc_info`：`ready: true` 且没有 `error` 才正常。
- 卡在 Building/Starting 或报 Runtime error：Settings → Restart this Space；仍不行看 Logs。
- 启动时报 `add the Space secret HF_TOKEN` 或下载 401/403：secret `HF_TOKEN` 失效（比如 token 被删了）。新建一个对私有仓库有读权限的 token，在 Settings → Variables and secrets 里替换 `HF_TOKEN`，Space 会自动重启。

## 已验证

| 检查 | 结果 |
|---|---|
| 上线自检（12 条已知答案） | 与已验证结果最大差 0.00001，判定 0 处不同 |
| 线上 194 条一份（一位说话人） | 103 秒；与模拟环境结果最大差 0.00001，判定 0 处不同 |
| 线上一次请求 999 条 | 573 秒后正常返回 —— HuggingFace 网关能撑住近 10 分钟的请求 |
| 线上整份 1,351 条（补丁 2026-10-05 版） | 1.8 秒收到响应头，890 秒打完 1,350 条（唯一被拒的是乱码文件名 `0_ncej [3. ncej-nchej].wav`）；旧版会被 1,000 条上限拒绝 |
| 线上新版确认 | 前端是新构建；响应体以保活空格开头；12 条与已验证结果最大差 0.00001、判定 0 处不同；录音与词数不符返回 400 + 说明 |
| 同镜像 2 CPU（CHTC） | 两个会话同时提交，结果各自与参考一致；整份会话中途断开后，下一份 11 秒完成（不取消要等约 15 分钟） |
| 模拟环境（同镜像、2 CPU） | 余弦相似度与原网站逐位相同；峰值内存 4.05 GB |

## 历史：fly.io（2026-10-04，已被 HuggingFace 取代，app 已于当天删除）

先部署在 https://hmong-audio.fly.dev（fly.io 个人 org yancyou，区域 ord，shared-cpu-2x 4 GB + 2 GB swap），上线自检通过，
但试用账号每台机器运行 5 分钟就被强制停止，必须绑卡；试用规格约 7 秒/条。fly 的部署文件（`fly.toml`、`deploy_to_fly.sh`）仍在
`~/hmong/web_deploy/`。部署中修过的 bug（端点被前端静态挂载截住返回 405）对 HuggingFace 同样适用，补丁里已修。
