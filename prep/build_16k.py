#!/usr/bin/env python3
"""Convert the Box word recordings to 16 kHz mono 16-bit WAV and write the pre-merge label index.

Recovered from the session log of 2026-08-24, the run that produced /staging/x/xouyang7/full32_16k.tar.
The conversion and naming logic is unchanged; only the paths became arguments. The default output goes
to build/ so a rerun never overwrites the canonical tar on staging.

Inputs : build/full32_parse.tsv (from prep/parse_labels.py)
         $HMONG_STAGING/hmong_segments/{index.tsv, shard-*.tar} (raw Box files)
Outputs: --out-tar   entries full32_16k/<speaker>/<n>_<word>.wav, numbered per speaker in (list_n, original path) order
         --out-index full32_index.csv as it was before the three-source label merge
Usage  : python3 prep/build_16k.py [--workers 6]
"""
import io, os, csv, tarfile, collections, time, argparse
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

STAGING = os.environ.get("HMONG_STAGING", "/staging/x/xouyang7")


def convert(args):
    """(name, raw bytes) -> (name, 16 kHz PCM16 WAV bytes, frames, error)."""
    name, blob = args
    try:
        wav, sr = sf.read(io.BytesIO(blob), dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            fr = Fraction(16000, sr); wav = resample_poly(wav, fr.numerator, fr.denominator)
        pk = np.abs(wav).max()
        if pk > 1.0: wav = wav / pk
        buf = io.BytesIO(); sf.write(buf, wav, 16000, subtype='PCM_16', format='WAV')
        return name, buf.getvalue(), len(wav), None
    except Exception as e:
        return name, None, 0, str(e)[:80]


def load_parse(path):
    with open(path) as f:
        return {r['orig']: r for r in csv.DictReader(f, delimiter='\t')}


def key_index(seg):
    m = {}
    for l in open(f"{seg}/index.tsv"):
        key, orig, size, shard, md5 = l.rstrip('\n').split('\t')
        m[orig] = (key, shard)
    return m


def plan_names(parse):
    """orig path -> <speaker>/<n>_<word>.wav, numbered per speaker in (list_n, orig) order."""
    per = collections.defaultdict(list)
    for orig, r in parse.items(): per[r['speaker']].append(r)
    plan = {}
    for spk, rs in per.items():
        rs.sort(key=lambda r: (int(r['list_n'] or 0), r['orig']))
        for i, r in enumerate(rs): plan[r['orig']] = f"{spk}/{i}_{r['word']}.wav"
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", default="build/full32_parse.tsv")
    ap.add_argument("--segments", default=f"{STAGING}/hmong_segments")
    ap.add_argument("--out-tar", default="build/full32_16k.tar")
    ap.add_argument("--out-index", default="build/full32_index_orig.csv")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    parse = load_parse(a.parse); kidx = key_index(a.segments); plan = plan_names(parse)
    by_shard = collections.defaultdict(list)
    for orig in plan:
        key, shard = kidx[orig]; by_shard[shard].append((key, orig))
    os.makedirs(os.path.dirname(a.out_tar) or ".", exist_ok=True)
    done = 0; errs = []; t0 = time.time(); out_rows = []
    with tarfile.open(a.out_tar, 'w') as out, ProcessPoolExecutor(a.workers) as ex:
        for shard in sorted(by_shard):
            tf = tarfile.open(f"{a.segments}/{shard}")
            want = {key: orig for key, orig in by_shard[shard]}
            batch = []
            for m in tf:
                stem = m.name[:-4] if m.name.endswith(('.wav', '.mp4')) else m.name
                if m.name.endswith('.wav') and stem in want:
                    batch.append((want[stem], tf.extractfile(m).read()))
            tf.close()
            for orig, data, frames, err in ex.map(convert, batch, chunksize=16):
                if err: errs.append((orig, err)); continue
                rel = plan[orig]
                ti = tarfile.TarInfo("full32_16k/" + rel); ti.size = len(data); ti.mtime = 0; ti.mode = 0o644
                out.addfile(ti, io.BytesIO(data))
                r = parse[orig]
                out_rows.append((rel, r['word'], r['label'], r['speaker'], r['realized'],
                                 r['note'], r['suspect'], frames, r['list_n'], orig))
                done += 1
            print(f"  {shard}: {done} files so far ({done/(time.time()-t0):.0f}/s)", flush=True)
    out_rows.sort()
    with open(a.out_index, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["wav_file", "word", "label", "speaker", "realized", "note", "suspect", "frames", "list_n", "orig_path"])
        for r in out_rows: w.writerow(r)
    print(f"\nDone: {done} files -> {a.out_tar} ({os.path.getsize(a.out_tar)/2**30:.2f} GiB), {len(errs)} failed")
    for o, e in errs[:5]: print("  ERR", o, e)


if __name__ == "__main__":
    main()
