#!/bin/bash
# 收取各 fold 的 checkpoint 到 /staging（选择性解压，避开 /tmp 容量限制）
mkdir -p /staging/x/xouyang7/full32_ckpts
SCR=/staging/x/xouyang7/.collect_scratch
for f in full32_*_rlz_output.tar.gz; do
  [ -f "$f" ] || continue
  FOLD=$(echo "$f" | sed 's/full32_\(.*\)_rlz_output.tar.gz/\1/')
  DST="/staging/x/xouyang7/full32_ckpts/full32_fold${FOLD}_best.pt"
  if [ -f "$DST" ]; then echo "已存在 fold${FOLD}, 跳过"; continue; fi
  echo "解压 fold${FOLD} ..."
  rm -rf "$SCR"; mkdir -p "$SCR"
  tar -xzf "$f" -C "$SCR" --wildcards '*checkpoint_best.pt' '*split.json' '*.log' 2>/dev/null
  B=$(find "$SCR" -name checkpoint_best.pt | head -1)
  if [ -z "$B" ]; then echo "✗ fold${FOLD} 内无 checkpoint_best.pt"; rm -rf "$SCR"; continue; fi
  mv "$B" "$DST"
  find "$SCR" -name split.json -exec mv {} "split_fold${FOLD}.json" \; 2>/dev/null
  find "$SCR" -name '*.log'   -exec mv {} "trainlog_fold${FOLD}.log" \; 2>/dev/null
  rm -rf "$SCR"
  echo "✓ fold${FOLD} -> $DST ($(du -h "$DST"|cut -f1))"
done
echo "--- ckpt 库 ---"; ls -lh /staging/x/xouyang7/full32_ckpts/
