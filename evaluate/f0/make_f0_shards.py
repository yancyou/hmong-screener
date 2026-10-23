#!/usr/bin/env python3
"""Split the labeled 32-speaker clips into 8 speaker-balanced shards for F0 feature extraction (one CHTC job per shard).
Recovered from the session log of 2026-08-24. Whole speakers stay in one shard because pitch is normalized per speaker.
Input : data/derived/full32_all_labels.csv     Output: build/f0/f0_shard{0-7}.csv, build/f0/f0_shard_list.txt
Usage : python3 evaluate/f0/make_f0_shards.py"""
import csv, collections, json, os
rows = list(csv.DictReader(open('data/derived/full32_all_labels.csv')))
by = collections.defaultdict(list)
for r in rows: by[r['speaker']].append(r)
spks = sorted(by, key=lambda s: -len(by[s]))
NS = 8
shards = [[] for _ in range(NS)]; load = [0] * NS
for s in spks:
    i = load.index(min(load)); shards[i].append(s); load[i] += len(by[s])
os.makedirs('build/f0', exist_ok=True)
for i, sh in enumerate(shards):
    with open(f'build/f0/f0_shard{i}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["wav_file", "word", "label", "speaker", "realized", "suspect"])
        for s in sh:
            for r in by[s]: w.writerow([r['wav_file'], r['word'], r['label'], r['speaker'], r['realized'], r['suspect']])
    print(f"shard{i}: {len(sh)} speakers / {load[i]} clips  {sh}")
json.dump({f"shard{i}": shards[i] for i in range(NS)}, open('build/f0/f0_shards.json', 'w'), indent=1)
open('build/f0/f0_shard_list.txt', 'w').write("".join(f"{i}\n" for i in range(NS)))
