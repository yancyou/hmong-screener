# 安装与本机测试

> **2026-10-05 更新**：模型不变。重新下载部署包、解压覆盖到软件目录、三个补丁重跑一遍（命令同下），重启后端；
> 改了界面，开着 `npm run dev` 会自动刷新，网站部署则要重新 `npm run build`。本次修正：
> 一次上传的上限从 1,000 条提高到 5,000 条（一整份词表约 1,350 条，以前会被拒绝）；
> 长会话（每条约 0.5 秒，1,000 条约 10 分钟）期间保持连接，浏览器不会中途断开；
> 两个会话同时提交时排队处理（以前会互相覆盖音高归一化）；关掉页面后服务器停止打分；界面直接显示服务器给的错误说明。
>
> **2026-09-28 更新（已经装过旧版的看这里）**：模型文件不变（仍是 `full32_foldfinal_infer.pt`，不用重新下载）。
> 重新下载 740 KB 的部署包，解压覆盖到软件目录，然后**三个补丁都重跑一遍**：
> ```bash
> tar -xzf ~/Downloads/ctc_deploy_kit.tar.gz
> python ctc_endpoint_patch.py && python gui_ctc_patch.py && python gui_table_patch.py
> ```
> 补丁会自动识别旧版并升级（重复运行安全）。然后重启后端；前端如果开着 `npm run dev` 会自动刷新。
> 本次修正：界面上的准确率改为复核后的数字；`tone ok` 列恢复显示概率（之前会把约两成正常录音标红）；
> 接口的音频读取与训练一致、加入会话内音高(F0)融合；部署文件缺失时只关闭 CTC 功能，不再拖垮整个后端。

## 一、需要下载的两样东西

```bash
# 部署套件（700 KB）
rsync -avP xouyang7@ap2002.chtc.wisc.edu:/staging/x/xouyang7/ctc_deploy_kit.tar.gz ~/Downloads/

# 模型（1.2 GB 推理版，已去掉训练用的优化器状态；与 3.6 GB 原版输出逐位一致）
rsync -avP xouyang7@ap2002.chtc.wisc.edu:/staging/x/xouyang7/full32_ckpts/full32_foldfinal_infer.pt ~/Downloads/
```

## 二、装进软件（3 条命令）

```bash
cd ~/Downloads/hmong_audio_software-main      # 你解压后的软件目录

tar -xzf ~/Downloads/ctc_deploy_kit.tar.gz    # 释放部署文件到当前目录
python ctc_endpoint_patch.py                  # 后端：加 /api/ctc_score 端点
python gui_ctc_patch.py && python gui_table_patch.py   # 前端：加打分方式切换
```

三个补丁都是**幂等**的，重复运行安全；都只做追加/替换，不删原功能。

## 三、装依赖

```bash
conda activate hmong-env

# 软件自带的 api_requirements.txt 缺 7 个包，必须补上，否则 server 起不来
pip install -r api_requirements.txt \
    pandas librosa editdistance pyarrow tensorboardX matplotlib pyyaml

cd GUI && npm install && cd ..     # 前端依赖（首次才需要）
```

## 四、启动

**终端 1 —— 后端**
```bash
cd ~/Downloads/hmong_audio_software-main
conda activate hmong-env

mkdir -p ctc_data && cp dict.ltr.txt ctc_data/
export DATA_DIR="$(pwd)/ctc_data"                        # ← 不设会导致打分全废
export CHECKPOINT_PATH=~/Downloads/full32_foldfinal_infer.pt

python -m uvicorn api_similarity_server:app --host 127.0.0.1 --port 8000
```

**终端 2 —— 前端**
```bash
cd ~/Downloads/hmong_audio_software-main/GUI
npm run dev
```

浏览器打开终端 2 显示的地址（通常 `http://localhost:5173`）。

## 五、验证装对了

### 1. 后端启动日志必须出现
```
Using dict_path: <你的目录>/ctc_data/dict.ltr.txt
```
如果是 `_fallback_data/dict.ltr.txt` → **字典错了**，所有录音会被判成可疑。

### 2. 命令行自测
```bash
curl http://localhost:8000/api/ctc_info
```
应返回 `{"ready": ..., "validated": {...}}`。

### 3. 界面自测
1. 打开 **Batch Score** 页
2. 顶部 **Scoring method** 选 `Closed-vocab CTC (+F0)`
3. 上传一次采集的 ZIP（Collect Data 页产出的）
4. 点 Run

**正确的表现**：绝大多数录音 `verdict = pass`（分数 0.9+），少数排在最前面标 `review`。
**如果全是 review 且分数接近 0** → 回去检查第 1 条（字典）。

## 六、界面怎么用

结果表按**可疑度从高到低**排序，第 1 行最可疑：

| 列 | 含义 |
|---|---|
| `#` | 复核顺序，从 1 开始往下看 |
| `word ok` | 目标词胜出概率，低 = 可能念成了别的词 |
| `tone ok` | 目标声调胜出概率（CTC 模型），低 = 声调可能错 |
| `verdict` | `pass` 自动放行 / `review` 需人工听 |

分数颜色：<0.3 红、0.3–0.7 黄、>0.7 正常。

**排序（`#` 列）综合了三路信号**：整词分、声调分、以及音高(F0)声调分。所以偶尔会有两列分数都不低、
却排在前面的录音 —— 那是音高判断它声调可疑。

**用法是"从上往下听，觉得后面都没问题了就停"**，不是看 verdict 下结论。
错误只占约 7%，标为 review 的里面仍有相当比例是正常发音。

**一次上传 = 一个说话人的一次录音**（Collect Data 页导出的 ZIP 就是这样）。音高要在同一个人的整场录音里做归一化，
混着上传多个人会让声调判断变差。

## 七、注意

- 换成 CTC 模型后，原来的 **Cosine similarity 模式会失效**（它需要另一个模型）。
  想两个都用需要跑两个 server 实例，各自指向不同 CHECKPOINT_PATH。
- CPU 就能跑：4 核实测 0.36 秒/条（含 F0 音高融合），350 词的 word list 约 2 分钟。

## 八、常见问题

- **Python 必须是 3.10**（软件 README 里的 `conda create -n hmong-env python=3.10` 就是这个版本）。
  用 3.9 启动后端会在导入 fairseq 时报 `TypeError: unsupported operand type(s) for |`。
- **CPU 打分异常慢或卡住**：多核共享机器上 torch 默认开满线程会互相抢占。启动后端前加
  `export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`（实测 12 条录音从数分钟降到 16 秒）。
- **`/api/ctc_info` 返回 `"ready": false` 并带 `error`**：部署包里的文件没放进软件目录（或缺依赖）。
  这时只有 CTC 功能被关闭，软件其它功能照常可用；按错误信息补齐文件后重启后端即可。
