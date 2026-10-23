#!/usr/bin/env python3
"""给 BatchScorePage.jsx 加 CTC 打分模式（幂等）。
保留原余弦模式，新增下拉切换 + CTC 专属列。"""
import re,sys,io
P="GUI/BatchScorePage.jsx"
VERSION="2026-10-05"
TAG=f"// ctc-gui-version: {VERSION}"
UPGRADES=[  # 旧版(2026-08)补丁留下的文字 → 新文字
    ('Validated on 32 speakers / 41,518 clips: word-error AUC 0.933, tone-error AUC 0.939.','Validated on 32 held-out speakers (41,518 clips): word-error AUC 0.92, tone-error AUC 0.88; pitch (F0) is fused into the ranking.'),
    ('errors are only ~6% of clips, so','errors are only ~7% of clips, so'),
    ('flagged items still need a human ear. Tone detection is weaker for some speakers.',"flagged items still need a human ear. Tone detection is weaker for some speakers. Upload one speaker's session at a time (pitch is normalized per session)."),
    ("'P_word', 'P_tone', 'P_f0_tone', 'suspicion', 'verdict', 'priority', 'error']","'P_word', 'P_tone', 'P_f0_tone', 'tone_fused', 'suspicion', 'verdict', 'priority', 'error']"),
    # 2026-10-05: 显示服务器给的错误说明 (CTC 长会话先回 200, 出错写在 JSON 的 error 里); 打分中提示要等多久; 验证规模更新为 33 人
    ("""      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();""",
     """      const text = await res.text();
      let data = null;
      try { data = JSON.parse(text); } catch (_) { data = null; }
      if (!res.ok || !data || (data.error && !(data.results && data.results.length))) {
        const msg = data && (data.error || data.detail);
        throw new Error(typeof msg === 'string' ? msg : `Server responded ${res.status}`);
      }"""),
    ('{scoring && <p className="text-xs text-gray-500 dark:text-slate-400 mt-3">First run builds a prototype per distinct word (cached afterward) — on CPU this can take a while for many words.</p>}',
     "{scoring && <p className=\"text-xs text-gray-500 dark:text-slate-400 mt-3\">{scoreMode === 'ctc' ? 'Scoring takes roughly half a second per recording (about 10 minutes for 1,000 recordings). Keep this tab open until the results appear.' : 'First run builds a prototype per distinct word (cached afterward) — on CPU this can take a while for many words.'}</p>}"),
    ('Validated on 32 held-out speakers (41,518 clips): word-error AUC 0.92, tone-error AUC 0.88','Validated on 33 held-out speakers (43,573 clips): word-error AUC 0.92, tone-error AUC 0.88'),
]
s=io.open(P,encoding="utf-8").read()
if TAG in s:
    print("GUI 补丁已是最新版，跳过"); sys.exit(0)
if "scoreMode" in s:                     # 装过旧版 → 定点升级
    n=0
    for a,b in UPGRADES:
        if a in s and b not in s: s=s.replace(a,b); n+=1   # b 已在 = 已升级过 (有的 b 包含 a)
    io.open(P,"w",encoding="utf-8").write(TAG+"\n"+s)
    print(f"✓ 检测到旧版 GUI 补丁，已升级 {n} 处"); sys.exit(0)

# 1) 状态：打分模式
s=s.replace("  const runBatch = async () => {",
"""  const [scoreMode, setScoreMode] = useState('ctc');   // 'ctc' | 'cosine'

  const runBatch = async () => {""",1)

# 2) 提交逻辑：按模式选端点
s=s.replace("""      const res = await fetch('/api/batch_score', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      // merge scores back onto metadata rows by filename
      const byName = new Map((data.results || []).map((x) => [x.filename, x]));
      const merged = rows.map((r) => ({ ...r, ...(byName.get(r.filename) || {}) }));
      setResults({ rows: merged, ...data });""",
"""      const endpoint = scoreMode === 'ctc' ? '/api/ctc_score' : '/api/batch_score';
      const res = await fetch(endpoint, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      const byName = new Map((data.results || []).map((x) => [x.filename, x]));
      let merged = rows.map((r) => ({ ...r, ...(byName.get(r.filename) || {}) }));
      if (scoreMode === 'ctc') {
        // 按可疑度升序：越靠前越需要人工复核
        merged = merged.slice().sort((a, b) => {
          const av = a.suspicion ?? a.P_joint ?? 2, bv = b.suspicion ?? b.P_joint ?? 2;
          return av - bv;
        });
      }
      setResults({ rows: merged, mode: scoreMode, ...data });""",1)

# 3) 通过数统计：CTC 模式看 verdict
s=s.replace("  const passCount = results ? results.rows.filter((r) => r.score != null && r.score >= threshold).length : 0;",
"""  const passCount = results
    ? (results.mode === 'ctc'
        ? results.rows.filter((r) => r.verdict === 'PASS').length
        : results.rows.filter((r) => r.score != null && r.score >= threshold).length)
    : 0;""",1)

# 4) CSV 导出列
s=s.replace("    const cols = ['filename', 'speaker_id', 'word', 'take', 'list_index', 'score', 'pass', 'in_library', 'n_templates', 'error'];",
"""    const cols = results.mode === 'ctc'
      ? ['filename', 'speaker_id', 'word', 'take', 'list_index', 'P_word', 'P_tone', 'P_f0_tone', 'tone_fused', 'suspicion', 'verdict', 'priority', 'error']
      : ['filename', 'speaker_id', 'word', 'take', 'list_index', 'score', 'pass', 'in_library', 'n_templates', 'error'];""",1)
s=s.replace("      lines.push(cols.map((c) => csvCell(c === 'pass' ? pass : (c === 'score' && r.score != null ? r.score.toFixed(4) : r[c]))).join(','));",
"""      lines.push(cols.map((c) => csvCell(
        c === 'pass' ? pass
        : (c === 'score' && r.score != null ? r.score.toFixed(4) : r[c])
      )).join(','));""",1)

# 5) UI：模式切换器（插在标题段之后）
s=s.replace("""        {/* Upload */}""",
"""        {/* 打分方式 */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">Scoring method</span>
            <select
              value={scoreMode}
              onChange={(e) => { setScoreMode(e.target.value); setResults(null); }}
              className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm"
            >
              <option value="ctc">Closed-vocab CTC (+F0)&nbsp;— recommended</option>
              <option value="cosine">Cosine similarity (legacy)</option>
            </select>
          </div>
          <p className="text-xs text-gray-600 dark:text-slate-400 mt-2 leading-relaxed">
            {scoreMode === 'ctc'
              ? 'Scores each recording against its target word and that word\\'s attested rivals (tone variants, aspiration pairs). No reference recording needed. Validated on 32 held-out speakers (41,518 clips): word-error AUC 0.92, tone-error AUC 0.88; pitch (F0) is fused into the ranking. Results are ordered most-suspicious first — review from the top.'
              : 'Cosine similarity between the recording and the reference-library prototype for its word. Requires the reference word audio library.'}
          </p>
          {scoreMode === 'ctc' && (
            <p className="text-xs text-amber-700 dark:text-amber-400 mt-2 leading-relaxed">
              Use this as a triage queue, not an automatic verdict: errors are only ~7% of clips, so
              flagged items still need a human ear. Tone detection is weaker for some speakers. Upload one speaker's session at a time (pitch is normalized per session).
            </p>
          )}
        </div>

        {/* Upload */}""",1)
for a,b in UPGRADES:                      # 新装也套用之后的修订
    if a in s and b not in s: s=s.replace(a,b)
io.open(P,"w",encoding="utf-8").write(TAG+"\n"+s)
print("✓ 已给 BatchScorePage.jsx 加 CTC 模式")
