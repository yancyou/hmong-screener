#!/bin/bash
mkdir -p /staging/x/xouyang7/full32v2_ckpts
SCR=/staging/x/xouyang7/.v2scratch
for f in full32v2_*_rlz_output.tar.gz; do
  [ -f "$f" ] || continue
  FOLD=$(echo "$f" | sed 's/full32v2_\(.*\)_rlz_output.tar.gz/\1/')
  DST="/staging/x/xouyang7/full32v2_ckpts/v2_fold${FOLD}_best.pt"
  if [ -f "$DST" ]; then echo "已存在 v2 fold${FOLD}"; rm -f "$f"; continue; fi
  echo "解压 v2 fold${FOLD} ..."
  rm -rf "$SCR"; mkdir -p "$SCR"
  tar -xzf "$f" -C "$SCR" --wildcards '*checkpoint_best.pt' '*split.json' '*.log' 2>/dev/null
  B=$(find "$SCR" -name checkpoint_best.pt | head -1)
  if [ -z "$B" ]; then echo "✗ fold${FOLD} 无 checkpoint"; rm -rf "$SCR"; continue; fi
  mv "$B" "$DST"
  find "$SCR" -name '*.log' -exec mv {} "trainlog_v2_fold${FOLD}.log" \; 2>/dev/null
  rm -rf "$SCR"
  SZ=$(stat -c %s "$DST"); REF=$(stat -c %s /staging/x/xouyang7/full32_ckpts/full32_fold0_best.pt)
  if [ "$SZ" = "$REF" ]; then rm -f "$f"; echo "✓ v2 fold${FOLD} 完整，已删 tarball"; 
  else echo "⚠️ v2 fold${FOLD} 大小异常 $SZ vs $REF，保留 tarball"; fi
done
ls -lh /staging/x/xouyang7/full32v2_ckpts/ 2>/dev/null
