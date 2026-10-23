#!/bin/bash
set -e
export INITIAL_DIR="$(pwd)"; echo "=== ASR fine-tune setup (fold=$1 variant=${2:-rlz}) ==="; date
tar -xzf fairseq.tar.gz
tar -xzf manifest_full32.tar.gz && echo "manifest_full32: $(ls manifest_full32|wc -l) dirs"
echo "staging -> full32_16k.tar"
cp /staging/x/xouyang7/full32_16k.tar . && tar -xf full32_16k.tar && rm full32_16k.tar
echo "full32_16k: $(find full32_16k -name '*.wav'|wc -l) wavs"
export PATH=$HOME/.local/bin:$PATH; export PYTHONPATH="${INITIAL_DIR}/fairseq:${PYTHONPATH}"
python -m pip install --user pip==24.0 2>&1|tail -1
python -m pip install --user -r requirements.txt 2>&1|tail -2
python -m pip install --user cython "numpy<2.0" "torch==2.1.0" --no-deps soundfile librosa editdistance 2>&1|tail -1
python -m pip install --user "setuptools<75" wheel 2>&1|tail -1
cd fairseq && python -m pip install --user --no-build-isolation -e . 2>&1|tail -2 && cd ..
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "=== train ==="; bash run_train.sh "$1" "${2:-rlz}"
echo "=== done ==="; date
