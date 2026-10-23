#!/usr/bin/env python3
"""Spot-check that prep/build_16k.py reproduces the canonical 16 kHz audio byte for byte.

Takes N recordings from one raw Box shard, converts them with build_16k.convert, and compares the bytes with the
same entries of $HMONG_STAGING/full32_16k.tar. Reading both tars takes a few minutes.
Usage (repo root): python3 tests/check_audio_sample.py [--shard shard-00003.tar] [--n 150]
"""
import os, sys, csv, random, tarfile, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prep"))
from build_16k import convert

ST = os.environ.get("HMONG_STAGING", "/staging/x/xouyang7")
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="shard-00003.tar"); ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()
rel = {r['orig_path']: r['wav_file'] for r in csv.DictReader(open("data/full32_merged.csv"))}
key2orig = {}
for l in open(f"{ST}/hmong_segments/index.tsv"):
    key, orig, size, shard, md5 = l.rstrip('\n').split('\t')
    if shard == a.shard and orig in rel: key2orig[key] = orig
random.seed(0); keys = set(random.sample(sorted(key2orig), min(a.n, len(key2orig))))
mine = {}
with tarfile.open(f"{ST}/hmong_segments/{a.shard}") as tf:
    for m in tf:
        stem = m.name[:-4]
        if m.name.endswith('.wav') and stem in keys:
            name, data, frames, err = convert((key2orig[stem], tf.extractfile(m).read()))
            mine["full32_16k/" + rel[name]] = data
want = set(mine); ref = {}
with tarfile.open(f"{ST}/full32_16k.tar") as tf:
    for m in tf:
        if m.name in want: ref[m.name] = tf.extractfile(m).read()
same = sum(mine[k] == ref.get(k) for k in mine)
print(f"{a.shard}: {same}/{len(mine)} converted files identical to full32_16k.tar")
sys.exit(0 if same == len(mine) and mine else 1)
