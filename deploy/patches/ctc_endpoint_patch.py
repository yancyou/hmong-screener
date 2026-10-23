#!/usr/bin/env python3
"""把闭集 CTC 打分端点追加到 api_similarity_server.py（幂等，可重复运行）"""
import re,sys
SRC="api_similarity_server.py"
MARK="# ==== CTC closed-vocab scoring (added) ===="
VERSION="2026-10-05"
s=open(SRC).read()
if MARK in s:
    if f"ctc-patch-version: {VERSION}" in s:
        print("补丁已是最新版，跳过"); sys.exit(0)
    s=s[:s.index(MARK)].rstrip()+"\n"          # 旧版补丁块在文件末尾 —— 整块替换
    open(SRC,"w").write(s); print("检测到旧版补丁，已移除，写入新版")
PATCH = '''

''' + MARK + '''
# ctc-patch-version: ''' + VERSION + '''
def _ctc_path(env, default):
    """数据文件默认放在 server 同目录 —— 不依赖从哪个目录启动"""
    v = os.environ.get(env, "").strip()
    return v if v else os.path.join(os.path.dirname(os.path.abspath(__file__)), default)

# CTC 功能缺文件/缺依赖时只关闭 CTC 端点，绝不拖垮 server 的其它功能（余弦相似度等）
try:
    from ctc_scorer import CTCScorer as _CTCScorer, F0_FUSION_WEIGHT
    _CTC = _CTCScorer(
        vocab_csv=_ctc_path("CTC_VOCAB", "full33_index.csv"),
        priors_json=_ctc_path("CTC_PRIORS", "priors_final.json"),
        thresholds_json=_ctc_path("CTC_THRESHOLDS", "deploy_thresholds.json"),
        dict_path=_ctc_path("CTC_DICT", "dict.ltr.txt"),
    )
    for _f in (_CTC.vocab_csv, _CTC.priors_json, _CTC.th_json):
        if not os.path.exists(_f):
            raise FileNotFoundError(_f)
    _CTC_UNAVAILABLE = None
except Exception as _e:
    _CTC = None
    _CTC_UNAVAILABLE = f"{type(_e).__name__}: {_e}"
    print(f"[ctc] CTC scoring disabled — {_CTC_UNAVAILABLE}")

# 双模型: 设了 CTC_CHECKPOINT_PATH 时, CTC 端点用它自己的模型, 主 CHECKPOINT_PATH 继续给余弦相似度用
# (网站: 余弦模型 + CTC 模型并存)。不设则与本地安装相同 —— CTC 复用 server 的主模型。
_CTC_OWN = None   # (model, task, normalize)
def _ctc_load_own(ckpt):
    """在模块导入时调用: 此时 server 还没接任何请求, 临时借用 server 自己的加载函数是安全的。"""
    global MODEL, TASK, NORMALIZE
    saved = (MODEL, TASK, NORMALIZE)
    env = {k: os.environ.get(k) for k in ("CHECKPOINT_PATH", "DATA_DIR", "W2V_PATH")}
    try:
        MODEL = None
        os.environ["CHECKPOINT_PATH"] = ckpt
        os.environ["DATA_DIR"] = _ctc_path("CTC_DATA_DIR", "ctc_data")   # 必须是含训练字典 dict.ltr.txt 的目录
        os.environ.pop("W2V_PATH", None)
        _load_model_locked()
        return (MODEL, TASK, NORMALIZE)
    finally:
        MODEL, TASK, NORMALIZE = saved
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

if _CTC is not None and os.environ.get("CTC_CHECKPOINT_PATH", "").strip():
    try:
        import time as _time
        _t0 = _time.time()
        _CTC_OWN = _ctc_load_own(os.environ["CTC_CHECKPOINT_PATH"].strip())
        print(f"[ctc] 独立 CTC 模型已加载 ({_time.time() - _t0:.0f}s): {os.environ['CTC_CHECKPOINT_PATH']}")
    except Exception as _e:
        _CTC = None
        _CTC_UNAVAILABLE = f"CTC model load failed — {type(_e).__name__}: {_e}"
        print(f"[ctc] CTC scoring disabled — {_CTC_UNAVAILABLE}")

def _ctc_model():
    """CTC 打分用的 (model, task, normalize)"""
    if _CTC_OWN is not None:
        return _CTC_OWN
    _load_model()
    return (MODEL, TASK, NORMALIZE)

def _ctc_read_wav(file):
    """CTC/F0 路径的音频读取 —— 与训练和交叉验证完全一致: 单声道 + 16 kHz(resample_poly)。
    不切静音、不在此做 layer_norm。(server 的 _read_wav_from_upload 会切首尾静音, 那是给余弦相似度用的;
    模型训练/评估都用未切的录音, 切静音会让线上分数偏离验证结果。)"""
    data = file.file.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(16000, sr)
        wav = resample_poly(wav, fr.numerator, fr.denominator)
    return np.ascontiguousarray(wav, dtype=np.float32)


_F0S = None
def _get_f0():
    """F0 声调打分器(惰性加载)。缺 f0_tone_clf.npz 或 librosa 时返回 None → 退回纯 CTC。"""
    global _F0S
    if _F0S is None:
        path = _ctc_path("CTC_F0_CLF", "f0_tone_clf.npz")
        try:
            from f0_infer import F0ToneScorer
            _F0S = F0ToneScorer(path) if os.path.exists(path) else False
        except Exception as e:
            print(f"[ctc] F0 融合不可用: {e}")
            _F0S = False
    return _F0S or None


_CTC_MAX_FILES = int(os.environ.get("CTC_MAX_FILES", "5000"))   # 一次上传的录音数上限 (框架默认 1000, 一整份词表会超过)
_CTC_SESSION_LOCK = threading.Lock()   # 会话串行: F0 说话人归一化是共享状态, 两个会话并行会互相覆盖; CPU 也只有一份


def _ctc_score_session(files, words, cancel=None):
    """一个会话(同一说话人的一份词表)的全部打分。在工作线程里同步执行; cancel 被置位(客户端断开)就提前停止。"""
    with _CTC_SESSION_LOCK:
        _m, _t, _nz = _ctc_model()
        F0S = _get_f0()
        rows, f0raw = [], []
        for f, w in zip(files, words):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled: the client disconnected")
            row = {"filename": f.filename, "word": (w or "").strip()}
            try:
                wav = _ctc_read_wav(f)
                row.update(_CTC.score(_m, _t, torch.from_numpy(wav), row["word"], _nz))
                if F0S is not None and row.get("P_joint") is not None:
                    try:
                        f0raw.append((len(rows), F0S.raw_logf0(wav, 16000)))
                    except Exception:
                        pass
            except Exception as e:
                row.update({"P_word": None, "P_tone": None, "P_joint": None,
                            "verdict": None, "error": str(e)})
            rows.append(row)
        ok = [r for r in rows if r.get("P_joint") is not None]
        fusion = "ctc only"
        if F0S is not None and len(f0raw) >= 10:
            # 说话人级归一化: 用本场全部录音估计音高基准 —— 与 F0 分类器训练时的特征一致
            F0S.fit_speaker([x[0] for _, x in f0raw], speaker="_session")
            for i, (lf, vr, mvp) in f0raw:
                rows[i]["P_f0_tone"] = round(float(F0S.posterior_from_logf0(lf, vr, mvp, rows[i]["word"], speaker="_session")[0]), 5)
            okf = [r for r in ok if r.get("P_f0_tone") is not None]
            if len(okf) >= 10:
                from scipy.stats import rankdata
                n = len(okf)
                rt = rankdata([r["P_tone"] for r in okf]) / n
                rf = rankdata([r["P_f0_tone"] for r in okf]) / n
                rw = rankdata([r["P_word"] for r in okf]) / n
                wt = F0_FUSION_WEIGHT
                for i, r in enumerate(okf):
                    r["tone_fused"] = round(float((1 - wt) * rt[i] + wt * rf[i]), 5)
                    r["suspicion"] = round(float(min(rw[i], r["tone_fused"])), 5)
                # 会话内按 suspicion 定档: 放行条数 = 本场按全局 P_joint 阈值会放行的条数, 放行最不可疑的那些
                n_pass = sum(1 for r in okf if r["P_joint"] >= _CTC.thr)
                for i, r in enumerate(sorted(okf, key=lambda r: -r["suspicion"])):
                    r["verdict"] = "PASS" if i < n_pass else "REVIEW"
                fusion = f"ctc+f0 (session, w={wt})"
        if fusion == "ctc only":
            ok.sort(key=lambda r: r["P_joint"])
        else:
            ok.sort(key=lambda r: r.get("suspicion", 2.0))
        for i, r in enumerate(ok):
            r["priority"] = i + 1 if r.get("verdict") == "REVIEW" else None
        n_rev = sum(1 for r in ok if r.get("verdict") == "REVIEW")
        return {
            "results": ok + [r for r in rows if r.get("P_joint") is None],
            "n_files": len(files),
            "n_scored": len(ok),
            "n_pass": len(ok) - n_rev,
            "n_review": n_rev,
            "threshold": _CTC.thr if _CTC._ready else None,
            "fusion": fusion,
            "method": "closed-vocab CTC (33-speaker CV calibrated, leak-free)",
        }


from starlette.requests import Request as _CtcRequest
from starlette.responses import StreamingResponse as _CtcStreamingResponse
from starlette.concurrency import run_in_threadpool as _ctc_run_in_threadpool
import asyncio as _ctc_asyncio
import json as _ctc_json


def _ctc_json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


@app.post("/api/ctc_score")
async def ctc_score(request: _CtcRequest):
    """闭集 CTC 发音评分（不需要参考录音）+ 会话内 F0 声调融合。

    表单: files (录音, 可多个) + words (目标词, 与 files 一一对应)。一次上传 = 一个会话（同一说话人的一份词表）。
    返回每条: P_word / P_tone / P_f0_tone / tone_fused / suspicion / P_joint / verdict / priority
    按 suspicion 升序 —— 越靠前越可疑，供人工从上往下复核。
    一整份词表要算好几分钟: 响应头立即返回, 打分期间每 10 秒发一个空格 (JSON 允许前导空白),
    免得浏览器 (如 Firefox 5 分钟收不到响应就断开) 或网关在等待中断开。
    """
    if _CTC is None:
        return JSONResponse(status_code=503, content={"error": f"CTC scoring unavailable: {_CTC_UNAVAILABLE}"})
    try:
        try:
            form = await request.form(max_files=_CTC_MAX_FILES, max_fields=_CTC_MAX_FILES + 100)
        except TypeError:   # 旧版 starlette 没有这两个参数
            form = await request.form()
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "error": f"Could not read the upload ({e}). At most {_CTC_MAX_FILES} recordings per session."})
    files = [f for f in form.getlist("files") if not isinstance(f, str)]
    words = [w if isinstance(w, str) else "" for w in form.getlist("words")]
    if not files or len(files) != len(words):
        await form.close()
        return JSONResponse(status_code=400, content={
            "error": f"Expected one word per recording: got {len(files)} recordings and {len(words)} words."})

    cancel = threading.Event()

    async def _body():
        job = _ctc_asyncio.ensure_future(_ctc_run_in_threadpool(_ctc_score_session, files, words, cancel))
        try:
            yield b" "
            while not job.done():
                done, _ = await _ctc_asyncio.wait({job}, timeout=10)
                if not done:
                    yield b" "
            try:
                out = job.result()
            except Exception as e:
                print(traceback.format_exc())
                out = {"error": str(e), "results": []}
            try:
                body = _ctc_json.dumps(out, ensure_ascii=False, allow_nan=False, default=_ctc_json_default)
            except Exception as e:
                body = _ctc_json.dumps({"error": f"Could not encode the results: {e}", "results": []})
            yield body.encode("utf-8")
        finally:
            cancel.set()   # 客户端中途断开时让工作线程尽快停下, 不占着锁和 CPU
            await form.close()

    return _CtcStreamingResponse(_body(), media_type="application/json",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/ctc_info")
def ctc_info():
    """CTC 评分器状态与已验证性能。"""
    if _CTC is None:
        return {"ready": False, "error": f"CTC scoring unavailable: {_CTC_UNAVAILABLE}"}
    return {
        "ready": bool(_CTC._ready),
        "n_candidates": int(_CTC.N) if _CTC._ready else None,
        "threshold": _CTC.thr if _CTC._ready else None,
        "validated": {
            "speakers": 33, "clips": 43573, "protocol": "4-fold leave-speakers-out, 按折无泄漏先验",
            "word_error_AUC": 0.918, "word_error_detect_at_FPR10": 0.830,
            "tone_error_AUC": 0.880, "tone_error_AUC_with_f0_per_session": 0.842,
            "word_recognition_top1": 0.928,
            "auto_pass_rate": 0.217, "auto_pass_correctness": 0.995,
            "auto_pass_rate_if_miss_le_10pct": 0.590,
        },
        "caveat": "指标对齐三源合并标注。声调检测在少数说话人上仍偏弱(融合后最差说话人 AUC 0.59)，建议按 priority 从上往下人工复核，不要仅依赖 PASS 自动放行。",
    }


# 前端静态文件 (GUI/dist) 挂在 "/" 上, 会截住所有在它之后注册的路由 (POST 返回 405)。
# 本补丁追加在文件末尾、挂载之后 —— 把挂载移回路由表末尾, 让 /api/ctc_* 生效。
from starlette.routing import Mount as _CtcMount
for _r in [r for r in app.router.routes if isinstance(r, _CtcMount)]:
    app.router.routes.remove(_r)
    app.router.routes.append(_r)
'''
open(SRC,"a").write(PATCH)
print(f"✓ 已追加 CTC 端点到 {SRC}")
