#!/bin/bash
set -e; set -o pipefail
export INITIAL_DIR="$(pwd)"; echo "=== 余弦模型嵌入提取 ==="; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
python -c "import zipfile;zipfile.ZipFile('/staging/x/xouyang7/hmong_audio_software-main.zip').extractall('.')"
cd hmong_audio_software-main
cp /staging/x/xouyang7/full32_16k.tar . && tar -xf full32_16k.tar && rm full32_16k.tar
cp /staging/x/xouyang7/full33_add_16k.tar . && tar -xf full33_add_16k.tar && rm full33_add_16k.tar
export PATH=$HOME/.local/bin:$PATH
echo "=== 安装依赖 ==="
python -m pip install --user pip==24.0 > ../pip0.log 2>&1 || { echo "⚠ pip 升级失败:"; tail -5 ../pip0.log; }
python -m pip install --user -r api_requirements.txt pandas librosa editdistance pyarrow tensorboardX matplotlib pyyaml > ../pip1.log 2>&1 || { echo "⚠ requirements 安装报错:"; grep -iE "error|failed" ../pip1.log | tail -8; }
python -m pip install --user --no-build-isolation -e ./fairseq > ../pip2.log 2>&1 || { echo "⚠ fairseq 可编辑安装报错 (改用 PYTHONPATH 加载):"; grep -iE "error|failed" ../pip2.log | tail -8; }
export PYTHONPATH="$(pwd)/fairseq:${PYTHONPATH}"
python -c "import fairseq, fastapi, librosa, soundfile, torch; print('imports OK | torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== 复制余弦模型 (staging 上的 HF 原文件副本) ==="; cp /staging/x/xouyang7/cosine_checkpoint_slim.pt checkpoint.pt && ls -l checkpoint.pt
export CHECKPOINT_PATH="$(pwd)/checkpoint.pt" EMBEDDING_LAYER=-1
echo "=== 提取嵌入 ==="; python ../cosine_embed.py --clips ../cmp_clips.csv --out ../cosine_emb.npz
date
