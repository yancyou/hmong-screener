#!/usr/bin/env python3
"""CTC 模式下的结果表格：显示 #, 词/调分数, 判定（幂等）"""
import io,sys
P="GUI/BatchScorePage.jsx"
s=io.open(P,encoding="utf-8").read()
VERSION="2026-09-28"
TAG=f"// ctc-table-version: {VERSION}"
UPGRADES=[('title="目标声调胜出概率(含F0融合)">tone ok</th>','title="目标声调胜出概率 (CTC)。排序和判定另外结合了音高 (F0)">tone ok</th>'),('const tone = r.tone_fused != null ? r.tone_fused : r.P_tone;','const tone = r.P_tone;   // 概率(0-1)才适用红/黄/正常着色; tone_fused 是会话内秩, 只用于排序')]
if TAG in s: print("表格补丁已是最新版，跳过"); sys.exit(0)
if "isCtc" in s:                         # 装过旧版 → 定点升级
    n=0
    for a,b in UPGRADES:
        if a in s: s=s.replace(a,b); n+=1
    io.open(P,"w",encoding="utf-8").write(TAG+"\n"+s)
    print(f"✓ 检测到旧版表格补丁，已升级 {n} 处"); sys.exit(0)

old_head="""                    <th className="py-2 pr-3 font-medium">word</th>
                    <th className="py-2 pr-3 font-medium">speaker</th>
                    <th className="py-2 pr-3 font-medium">take</th>
                    {!simpleView && <th className="py-2 pr-3 font-medium">score</th>}
                    <th className="py-2 pr-3 font-medium">{simpleView ? 'result' : 'pass'}</th>
                    <th className="py-2 pr-3 font-medium">note</th>"""
new_head="""                    {isCtc && <th className="py-2 pr-3 font-medium">#</th>}
                    <th className="py-2 pr-3 font-medium">word</th>
                    <th className="py-2 pr-3 font-medium">speaker</th>
                    <th className="py-2 pr-3 font-medium">take</th>
                    {isCtc && !simpleView && <th className="py-2 pr-3 font-medium" title="目标词胜出概率">word ok</th>}
                    {isCtc && !simpleView && <th className="py-2 pr-3 font-medium" title="目标声调胜出概率 (CTC)。排序和判定另外结合了音高 (F0)">tone ok</th>}
                    {!isCtc && !simpleView && <th className="py-2 pr-3 font-medium">score</th>}
                    <th className="py-2 pr-3 font-medium">{simpleView ? 'result' : (isCtc ? 'verdict' : 'pass')}</th>
                    <th className="py-2 pr-3 font-medium">note</th>"""
assert old_head in s; s=s.replace(old_head,new_head,1)

old_row="""                  {results.rows.map((r, i) => {
                    const pass = r.score != null && r.score >= threshold;
                    return (
                      <tr key={i} className="border-b border-slate-100 dark:border-slate-800/60">
                        <td className="py-2 pr-3 font-medium text-slate-800 dark:text-slate-100">{r.word}</td>
                        <td className="py-2 pr-3 text-slate-500 dark:text-slate-400">{r.speaker_id || ''}</td>
                        <td className="py-2 pr-3 text-slate-500 dark:text-slate-400 tabular-nums">{r.take || ''}</td>
                        {!simpleView && <td className="py-2 pr-3 tabular-nums font-semibold text-slate-900 dark:text-white">{r.score != null ? r.score.toFixed(3) : '—'}</td>}
                        <td className="py-2 pr-3">{r.score == null ? '' : (
                          <span className={cn('px-2 py-0.5 rounded text-xs font-medium', pass ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300')}>
                            {simpleView ? (pass ? 'Correct' : 'Not correct') : (pass ? 'pass' : 'below')}
                          </span>
                        )}</td>
                        <td className="py-2 pr-3 text-xs text-amber-600 dark:text-amber-400">{r.in_library === false ? 'not in library' : (r.error || '')}</td>
                      </tr>
                    );"""
new_row="""                  {results.rows.map((r, i) => {
                    const pass = isCtc ? (r.verdict === 'PASS') : (r.score != null && r.score >= threshold);
                    const scored = isCtc ? (r.P_word != null) : (r.score != null);
                    const tone = r.P_tone;   // 概率(0-1)才适用红/黄/正常着色; tone_fused 是会话内秩, 只用于排序
                    const dim = (v) => v == null ? '—' : (v < 0.3 ? 'text-red-600 dark:text-red-400 font-semibold'
                                   : v < 0.7 ? 'text-amber-600 dark:text-amber-400' : 'text-slate-700 dark:text-slate-200');
                    return (
                      <tr key={i} className="border-b border-slate-100 dark:border-slate-800/60">
                        {isCtc && <td className="py-2 pr-3 text-slate-400 tabular-nums">{i + 1}</td>}
                        <td className="py-2 pr-3 font-medium text-slate-800 dark:text-slate-100">{r.word}</td>
                        <td className="py-2 pr-3 text-slate-500 dark:text-slate-400">{r.speaker_id || ''}</td>
                        <td className="py-2 pr-3 text-slate-500 dark:text-slate-400 tabular-nums">{r.take || ''}</td>
                        {isCtc && !simpleView && <td className={cn('py-2 pr-3 tabular-nums', dim(r.P_word))}>{r.P_word != null ? r.P_word.toFixed(3) : '—'}</td>}
                        {isCtc && !simpleView && <td className={cn('py-2 pr-3 tabular-nums', dim(tone))}>{tone != null ? Number(tone).toFixed(3) : '—'}</td>}
                        {!isCtc && !simpleView && <td className="py-2 pr-3 tabular-nums font-semibold text-slate-900 dark:text-white">{r.score != null ? r.score.toFixed(3) : '—'}</td>}
                        <td className="py-2 pr-3">{!scored ? '' : (
                          <span className={cn('px-2 py-0.5 rounded text-xs font-medium', pass ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300')}>
                            {simpleView ? (pass ? 'Correct' : 'Needs review') : (isCtc ? (pass ? 'pass' : 'review') : (pass ? 'pass' : 'below'))}
                          </span>
                        )}</td>
                        <td className="py-2 pr-3 text-xs text-amber-600 dark:text-amber-400">{r.in_library === false ? 'not in library' : (r.error || '')}</td>
                      </tr>
                    );"""
assert old_row in s; s=s.replace(old_row,new_row,1)
# isCtc 定义
s=s.replace("  const passCount = results","  const isCtc = results ? results.mode === 'ctc' : (scoreMode === 'ctc');\n  const passCount = results",1)
io.open(P,"w",encoding="utf-8").write(TAG+"\n"+s)
print("✓ 表格已支持 CTC 模式（序号/词分/调分/verdict）")
