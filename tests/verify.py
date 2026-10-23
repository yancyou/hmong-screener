#!/usr/bin/env python3
"""Reproducibility and sanity checks for the repository (CPU only; run on the CHTC access point from any folder).

Quick checks (about 2 minutes):
  static code checks, secret scan, regenerated manifests and priors versus the recorded checksums,
  the deployed model's training split, and a rebuild of the deploy kit versus the copy on staging.
--full adds the slow checks (about 35 minutes):
  label re-parse from the raw Box file names, a 150-file audio conversion spot-check, a rebuild of the website
  bundle versus the deployed bundle (needs npm on PATH), and reruns of select_model.py / eval_combos.py.
--live also checks the public website with 12 recordings of known result.
Everything except the static checks and the secret scan needs the private files and CHTC staging (README,
"Private files"); without them those checks are skipped.
Usage: python3 tests/verify.py [--full] [--live]
"""
import argparse, ast, glob, hashlib, json, os, re, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ST = os.environ.get("HMONG_STAGING", "/staging/x/xouyang7")
PRIVATE = os.path.exists("data/speakers.json") and os.path.isdir(ST)
SCRUBBED = {"DEPLOY_WEB.md"}   # kit document edited after the kit was built: a speaker code removed for the public repository
results = []


def check(name, fn, private=True):
    if private and not PRIVATE:
        print(f"SKIP  {name}  needs the private files and CHTC staging", flush=True)
        return
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


def tar_hashes(path):
    out = {}
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile():
                h = hashlib.sha256(); f = t.extractfile(m)
                for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
                out[m.name] = h.hexdigest()
    return out


def compare_tars(new, old):
    """Rebuilt archive versus deployed one. Python files whose code is unchanged (only comments differ) and
    the documents in SCRUBBED count as matching; everything else must be byte-identical."""
    a, b = tar_hashes(new), tar_hashes(old)
    diff = [k for k in a if a[k] != b.get(k)]

    def read(path, name):
        with tarfile.open(path) as t: return t.extractfile(name).read()
    comments = [k for k in diff if k.endswith(".py") and k in b
                and ast.dump(ast.parse(read(new, k))) == ast.dump(ast.parse(read(old, k)))]
    edited = [k for k in diff if os.path.basename(k) in SCRUBBED]
    bad = [k for k in diff if k not in comments and k not in edited] + [k for k in b if k not in a]
    detail = f"{len(a) - len(diff)}/{len(b)} files identical to {old}"
    if comments: detail += f"; comments edited since: {comments}"
    if edited: detail += f"; speaker codes removed since: {edited}"
    if bad: detail += f"; different: {bad[:5]}"
    return not bad, detail


def code_files(ext):
    return [p for p in glob.glob(f"**/*{ext}", recursive=True) if not p.startswith("build/")]


# ---------------------------------------------------------------- quick checks
def static_python():
    bad = [p for p in code_files(".py") if run(f"python3 -m py_compile '{p}'").returncode]
    run("find . -name __pycache__ -not -path './build/*' -exec rm -rf {} +")
    return not bad, f"{len(code_files('.py'))} files" + (f", failing: {bad}" if bad else "")


def static_shell():
    bad = [p for p in code_files(".sh") if run(f"bash -n '{p}'").returncode]
    return not bad, f"{len(code_files('.sh'))} files" + (f", failing: {bad}" if bad else "")


def patch_payload():
    src = open("deploy/patches/ctc_endpoint_patch.py").read()
    ns = {}
    code = src.split('open(SRC,"a").write(PATCH)')[0].replace('s=open(SRC).read()', 's=""').replace("sys.exit(0)", "pass")
    exec(compile(code, "patch", "exec"), ns)
    ast.parse(ns["PATCH"])
    return True, f"server code added by the patch parses (version {ns['VERSION']})"


def secrets():
    pat = re.compile(r"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{20,}|FlyV1 |BEGIN (RSA|OPENSSH) PRIVATE")
    hits = [p for p in code_files(".py") + code_files(".sh") + code_files(".md") + code_files(".sub") + code_files(".txt")
            if p != "tests/verify.py" and pat.search(open(p, errors="ignore").read())]   # this file holds the patterns
    return not hits, "no tokens or keys in code and docs" if not hits else f"found in {hits}"


def speaker_codes():
    """No participant speaker code in any file git tracks (the files that get published)."""
    codes = sorted({s for f in json.load(open("data/folds33.json")).values() for s in f["test"]})
    pat = re.compile(r"(?i)(?<![A-Za-z0-9])(" + "|".join(map(re.escape, codes)) + r")(?![0-9])")
    tracked = run("git ls-files").stdout.split()
    hits = [p for p in tracked if not p.endswith((".wav", ".npz", ".npy", ".gz", ".pt"))
            and os.path.exists(p) and pat.search(open(p, errors="ignore").read())]
    return not hits, f"none of the {len(codes)} speaker codes in {len(tracked)} tracked files" if not hits else f"found in {hits}"


def manifests_and_priors():
    r = run("bash prep/make_job_tarballs.sh && python3 prep/make_manifests.py --index data/full32_index.csv "
            "--folds data/folds.json --out build/manifest_full32 && python3 prep/make_priors.py")
    if r.returncode: return False, r.stderr[-300:]
    exp = [l.split("  ", 1) for l in open("tests/expected/generated.sha256").read().splitlines() if l]
    bad = [p for h, p in exp if not os.path.exists(p) or sha(p) != h]
    return not bad, f"{len(exp) - len(bad)}/{len(exp)} regenerated files match the recorded checksums" + (f"; first mismatches {bad[:3]}" if bad else "")


def deployed_split():
    s = json.load(open("build/manifest_v1/final_rlz/split.json")); ref = json.load(open("data/split_foldfinal.json"))
    keys = ["test", "valid", "train", "variant", "n_train", "n_valid", "n_train_realized"]
    same_dict = open("build/manifest_v1/dict.ltr.txt").read() == open("deploy/assets/dict.ltr.txt").read()
    ok = all(s[k] == ref[k] for k in keys) and same_dict
    return ok, f"train {s['n_train']} / valid {s['n_valid']} clips, dictionary matches the model: {same_dict}"


def kit():
    r = run("bash deploy/build_kit.sh build/verify_kit.tar.gz")
    if r.returncode: return False, r.stderr[-300:]
    return compare_tars("build/verify_kit.tar.gz", f"{ST}/ctc_deploy_kit.tar.gz")


# ---------------------------------------------------------------- slow checks
def labels_reparse():
    r = run("python3 prep/parse_labels.py")
    if r.returncode: return False, r.stderr[-300:]
    import csv
    P = {x['orig']: x for x in csv.DictReader(open("build/full32_parse.tsv"), delimiter='\t')}
    diff = 0; n = 0
    for x in csv.DictReader(open("data/full32_merged.csv")):
        p = P.get(x['orig_path']); n += 1
        if p is None or (x['word'], x['label'], x['realized'], x['note'], x['suspect'], x['list_n']) != \
                (p['word'], p['label'], p['realized'], p['note'], p['suspect'], p['list_n']): diff += 1
    folds = json.load(open("build/folds.json")) == json.load(open("data/folds.json"))
    pri = open("build/confusion_priors.json").read() == open("data/confusion_priors.json").read()
    return diff == 0 and folds and pri, f"{n} clips, {diff} label differences; folds identical {folds}; priors identical {pri}"


def audio_sample():
    r = run("python3 tests/check_audio_sample.py --shard shard-00003.tar --n 150")
    return r.returncode == 0, (r.stdout.strip() or r.stderr[-300:])


def bundle():
    out = "build/hf_bundle_verify.tar"   # local disk: long reads from staging have hung
    r = run(f"bash deploy/website/build_bundle.sh {out}")
    if r.returncode: return False, (r.stdout + r.stderr)[-300:]
    return compare_tars(out, f"{ST}/hf_bundle.tar")


def eval_reruns():
    msgs = []; ok = True
    for script, name in (("evaluate/select_model.py", "select_model.json"), ("evaluate/eval_combos.py", "eval_combos.json")):
        r = run(f"python3 {script}")
        same = r.returncode == 0 and json.load(open(f"build/{name}")) == json.load(open(f"tests/expected/{name}"))
        ok &= same; msgs.append(f"{name} identical: {same}")
    return ok, "; ".join(msgs)


def live():
    r = run("python3 deploy/website/smoke/live_check.py https://youngxuan-hmong-screener.hf.space")
    return r.returncode == 0, r.stdout.strip().splitlines()[-2] if r.stdout.strip() else r.stderr[-300:]


ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--live", action="store_true")
a = ap.parse_args()
os.makedirs("build", exist_ok=True)
for name, fn in (("python syntax", static_python), ("shell syntax", static_shell), ("server patch payload", patch_payload),
                 ("secret scan", secrets)):
    check(name, fn, private=False)
for name, fn in (("speaker codes", speaker_codes), ("manifests and priors", manifests_and_priors),
                 ("deployed model's split", deployed_split), ("deploy kit rebuild", kit)):
    check(name, fn)
if a.full:
    for name, fn in (("label re-parse", labels_reparse), ("audio spot-check", audio_sample),
                     ("website bundle rebuild", bundle), ("evaluation reruns", eval_reruns)):
        check(name, fn)
if a.live:
    check("live website", live)
print(f"\n{sum(results)}/{len(results)} checks passed" + ("" if PRIVATE else " (others skipped: no private files)"))
sys.exit(0 if all(results) else 1)
