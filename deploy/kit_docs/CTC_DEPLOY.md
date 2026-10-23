# 闭集 CTC 发音评分 —— 部署说明

在原有余弦相似度之外，新增一套**不需要参考录音**的评分方法。
原有功能完全不受影响。

## 新增文件

| 文件 | 说明 |
|---|---|
| `ctc_scorer.py` | 评分器核心（候选全集与离线评估完全一致） |
| `ctc_endpoint_patch.py` | 把端点追加进 server（幂等；检测到旧版会自动替换） |
| `f0_infer.py` + `f0_tone_clf.npz` | F0 音高声调打分器，与 CTC 声调分在会话内融合 |
| `full33_index.csv` | 词表（1,225 词） |
| `priors_final.json` | 实测混淆先验（33 人，827 个词对） |
| `deploy_thresholds.json` | 33 人四折交叉验证（无泄漏）标定的阈值 |

## 需要的模型

**不能用原来的 `checkpoint_last_1.pt`** —— 那是为余弦相似度训练的。CTC 评分需要:

```
full32_foldfinal_infer.pt   (1.2 GB, 已去掉训练用的优化器状态; 与 3.6 GB 原版输出逐位一致)
位置: /staging/x/xouyang7/full32_ckpts/full32_foldfinal_infer.pt
```

## 启动

见 `INSTALL_AND_TEST.md`（Python 3.10；必须设 `DATA_DIR` 指向含 `dict.ltr.txt` 的目录）。
纯 CPU 即可：4 核实测 0.36 秒/条（含 F0 音高提取），一份 350 词的 word list 约 2 分钟。

## 接口

### `POST /api/ctc_score`
输入: `files[]`(录音) + `words[]`(对应目标词)。**一次上传 = 一个会话**（同一说话人的一份词表）。

音频预处理与训练完全一致：单声道 + 16 kHz，**不切首尾静音**
（server 给余弦相似度用的切静音会让 CTC 分数偏离验证结果，此端点绕开它）。

返回（按 `suspicion` 升序，第 1 条最可疑）:
```json
{
  "results": [
    {"filename":"a2.wav","word":"tsheb","P_word":0.031,"P_tone":0.887,"P_f0_tone":0.42,
     "tone_fused":0.51,"suspicion":0.03,"P_joint":0.027,"verdict":"REVIEW","priority":1},
    {"filename":"a1.wav","word":"hlub","P_word":0.998,"P_tone":0.995,"P_f0_tone":0.91,
     "tone_fused":0.93,"suspicion":0.93,"P_joint":0.993,"verdict":"PASS","priority":null}
  ],
  "n_pass": 1, "n_review": 1, "fusion": "ctc+f0 (session, w=0.4)"
}
```

- `P_word` 目标词在其对手集中胜出的概率 —— 判**整词**对错
- `P_tone` CTC 给出的目标声调胜出概率；`P_f0_tone` F0 音高分类器给出的目标声调概率
- `tone_fused` 两路声调在**本场会话内**的秩融合（CTC 0.6 + F0 0.4）；F0 用本场全部录音做说话人音高归一化（与训练一致）
- `suspicion` = min(整词秩, 融合声调秩)，越低越可疑；`priority` REVIEW 项的复核顺序
- 本场少于 10 条可用录音时不做融合（`fusion: "ctc only"`），按 `P_joint` 排序
- 界面的 `tone ok` 列显示 `P_tone`（概率）；`tone_fused` 是会话内秩，只用于排序，不适合按 0.3/0.7 着色
- **长会话**：响应头立即返回，打分期间每 10 秒发一个空格保持连接（JSON 允许前导空白，浏览器的 `res.json()` 和 Python 的 `json.loads` 照常解析）。
  请求本身有问题（录音和词数不符、超过上限）返回 400 + `{"error": ...}`；开始打分后才出的错，HTTP 仍是 200，错误写在 JSON 的 `error` 里（界面会显示）
- 一次最多 5,000 条录音（环境变量 `CTC_MAX_FILES`）；同时提交的会话排队依次处理（F0 会话内归一化是共享状态）；客户端断开后服务器停止该会话的打分
- 部署文件缺失或导入失败时，CTC 端点返回 503（`/api/ctc_info` 返回 `ready: false` 和原因），server 其它功能不受影响
- 模型字典会与部署包 `dict.ltr.txt` 逐位核对，不一致时直接报错（防止 DATA_DIR 没设时"所有录音都被判可疑"的静默错误）
- **与余弦模型并存（网站）**：设 `CTC_CHECKPOINT_PATH=<CTC 模型>`，CTC 端点就加载自己的模型（字典目录 `CTC_DATA_DIR`，默认 `./ctc_data`），
  主 `CHECKPOINT_PATH` 继续给余弦相似度用。网站部署步骤见 `DEPLOY_WEB.md`

### `GET /api/ctc_info`
返回评分器状态和已验证性能指标。

## 已验证性能（四折留出说话人交叉验证，按折无泄漏先验，2026-09-22）

| 指标 | 数值 |
|---|---|
| 认词 top-1（词基） | 0.927 |
| 整词错检测 AUC | **0.921**（排除单源孤证标注 0.937） |
| 声调错检测 AUC | 0.880（CTC）；会话内 +F0 融合 按说话人平均 0.842 |
| 自动放行（放行内正确率 ≥99.5%，当前阈值 P_joint ≥ 0.994） | 21.7%（33 人 43,573 条上标定） |
| 自动放行（若改为允许漏检 10%，P_joint ≥ 0.780） | 59.0%（放行内正确率 98.8%） |

部署端打分与离线评估逐条一致（同模型同录音最大差 < 1e-5）。

## ⚠️ 使用限制

**声调检测在部分说话人上会失效** —— 全局阈值下漏检率中位数 12.7%，最差的说话人 72.7%
（排除不可靠的单源孤证标注后：中位 6.3%，最差 46.2%），且模型对此毫无察觉（自信地判错）。

因此建议：**按 `priority` 从上往下做人工复核，不要仅依赖 `PASS` 自动放行。**
把它当作"把人工注意力引导到可疑录音上"的排序工具，而不是自动判定器。

## ⚠️ 软件自身的依赖缺失（实测发现）

`api_requirements.txt` **漏了 `pandas`** —— 仓库自带的 fairseq 里
`fairseq/data/audio/gender_contrastive_dataset.py` 会 `import pandas`，
而 fairseq 启动时会遍历导入所有 task，因此缺 pandas 会导致 **server 完全起不来**。

**实测扫描了 93 个会被 fairseq 自动导入的文件，共 7 个必需依赖未声明**：

```bash
pip install -r api_requirements.txt \
    pandas librosa editdistance pyarrow tensorboardX matplotlib pyyaml
```

fairseq 启动时会遍历导入 `tasks/`、`criterions/`、`data/audio/` 下的所有模块，
任何一个缺依赖都会让 server 完全起不来，且报错信息指向不相关的 task 名，
排查困难。本地 conda 环境能跑是因为这些包碰巧已装。
