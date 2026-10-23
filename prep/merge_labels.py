#!/usr/bin/env python3
"""Merge three label sources into data/full32_merged.csv (label_merged, src), then write the v2 training index.

Recovered from the session log of 2026-08-25, the run that produced data/full32_merged.csv and data/full32_index.csv.
The voting logic is unchanged; only the paths became arguments, and outputs default to build/.

Sources, matched by (speaker, word, occurrence number):
  current : the file-name notes, i.e. the pre-merge index from prep/build_16k.py (`label` column)
  xlsx    : the annotators' Excel sheets, flattened to template_manifest_all_xlsx_speakers.csv
  june    : an earlier labeling pass, data/sources/new16k_index.csv
Rules: all sources say correct -> agree-ok; all say error -> agree-err (word error wins over tone error);
       sources disagree -> union-err (any error counts, the least reliable label); only the current source -> cur-only;
       label 0 (uncertain) stays 0 with src = cur.

KNOWN GAP: template_manifest_all_xlsx_speakers.csv was not kept, so this script cannot be rerun as is.
The merged result is preserved in data/full32_merged.csv, which keeps both the pre-merge `label` and `label_merged`.
Usage: python3 prep/merge_labels.py --xlsx <template_manifest_all_xlsx_speakers.csv>
"""
import csv, collections, argparse, os, sys


def norm(w): return (w or '').strip().lower()


def load(path, scol='speaker'):
    per = collections.defaultdict(int); out = {}
    for r in csv.DictReader(open(path)):
        l = (r.get('label') or '').strip()
        if l not in ('1', '2', '3', '4'): continue
        s = norm(r.get(scol)); w = norm(r.get('word'))
        if not s or not w: continue
        k = (s, w, per[(s, w)]); per[(s, w)] += 1
        out[k] = int(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default="build/full32_index_orig.csv", help="pre-merge index (prep/build_16k.py)")
    ap.add_argument("--xlsx", default="data/sources/template_manifest_all_xlsx_speakers.csv")
    ap.add_argument("--june", default="data/sources/new16k_index.csv")
    ap.add_argument("--out-merged", default="build/full32_merged.csv")
    ap.add_argument("--out-index", default="build/full32_index.csv", help="v2 training index (label := label_merged)")
    a = ap.parse_args()
    for p in (a.current, a.xlsx, a.june):
        if not os.path.exists(p): sys.exit(f"missing input {p} (see KNOWN GAP in this file's docstring)")
    A = load(a.xlsx); B = load(a.june)
    per = collections.defaultdict(int); rows = []
    for r in csv.DictReader(open(a.current)):
        if r['label'] not in ('1', '2', '3'):
            r['_k'] = None; rows.append(r); continue
        s = r['speaker']; w = r['word']
        r['_k'] = (s, w, per[(s, w)]); per[(s, w)] += 1; rows.append(r)
    stat = collections.Counter()
    for r in rows:
        k = r.get('_k')
        if k is None: r['label_merged'] = r['label']; r['src'] = 'cur'; continue
        cur = int(r['label']); votes = [cur]
        for X in (A, B):
            if k in X: votes.append(X[k])
        if len(votes) == 1:
            r['label_merged'] = str(cur); r['src'] = 'cur-only'; stat['current only'] += 1; continue
        err = [v for v in votes if v != 1]
        if not err:
            r['label_merged'] = '1'; r['src'] = 'agree-ok'; stat['agree correct'] += 1
        elif all(v != 1 for v in votes):
            lbl = 3 if any(v in (3, 4) for v in votes) else 2
            r['label_merged'] = str(lbl); r['src'] = 'agree-err'; stat['agree error'] += 1
        else:
            lbl = 3 if any(v in (3, 4) for v in err) else 2
            r['label_merged'] = str(lbl); r['src'] = 'union-err'; stat['disagree -> error'] += 1
    os.makedirs(os.path.dirname(a.out_merged) or ".", exist_ok=True)
    cols = [c for c in rows[0].keys() if c != '_k']
    with open(a.out_merged, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in cols})
    # v2 training index: merged label; new errors without a realized transcript are kept out of transcript training
    for r in rows:
        r['label'] = r['label_merged']
        if r['label'] in ('2', '3') and not r.get('realized'):
            r['realized'] = ''; r['suspect'] = '1'
    with open(a.out_index, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, '') for c in cols})
    print("merge:", dict(stat))


if __name__ == "__main__":
    main()
