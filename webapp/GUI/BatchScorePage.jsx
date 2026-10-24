import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Upload, Waves, Loader2, Download, FolderUp, FileArchive, Info } from 'lucide-react';
import { cn, Card, Button, Badge, PageTabs } from './ui.jsx';
import { readZip } from './zip.js';

const basename = (p) => (p || '').split('/').pop();
const csvCell = (v) => {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

// Tiny RFC-4180-ish CSV parser (handles quoted cells).
function parseCSV(text) {
  const rows = [];
  let i = 0, field = '', row = [], inQ = false;
  while (i < text.length) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; continue; } inQ = false; i++; continue; }
      field += c; i++; continue;
    }
    if (c === '"') { inQ = true; i++; continue; }
    if (c === ',') { row.push(field); field = ''; i++; continue; }
    if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = []; i++; continue;
    }
    field += c; i++;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function rowsToObjects(rows) {
  if (!rows.length) return [];
  const header = rows[0].map((h) => h.trim());
  return rows.slice(1).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}

export default function BatchScorePage({ page = 'batch', onNavigate }) {
  const [dark, setDark] = useState(() => (typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false));
  const toggleDark = () => {
    const next = !dark; setDark(next);
    document.documentElement.classList.toggle('dark', next);
    try { localStorage.setItem('theme', next ? 'dark' : 'light'); } catch {}
  };
  useEffect(() => {
    try { const s = localStorage.getItem('theme'); if (s) { const d = s === 'dark'; setDark(d); document.documentElement.classList.toggle('dark', d); } } catch {}
  }, []);

  const [embLayer, setEmbLayer] = useState('');
  const [threshold, setThreshold] = useState(0.70);
  // Audiologists reviewing a session mostly want the verdict, not the cosine
  // value behind it; the numbers stay in the CSV either way.
  const [simpleView, setSimpleView] = useState(false);
  const [numLayers, setNumLayers] = useState(null);

  // Loaded session: metadata rows + a map basename -> Blob
  const [meta, setMeta] = useState([]);        // metadata rows (objects)
  const [audio, setAudio] = useState(null);    // Map basename -> Blob
  const [sourceName, setSourceName] = useState('');
  const [loadErr, setLoadErr] = useState('');
  const [loading, setLoading] = useState(false);

  const [scoring, setScoring] = useState(false);
  const [results, setResults] = useState(null); // { rows, mean, ... }
  const [error, setError] = useState('');

  const zipInput = useRef(null);
  const dirInput = useRef(null);
  useEffect(() => { if (dirInput.current) { dirInput.current.setAttribute('webkitdirectory', ''); dirInput.current.setAttribute('directory', ''); } }, []);

  useEffect(() => {
    fetch('/api/model_info').then((r) => r.ok ? r.json() : null).then((d) => {
      if (d && typeof d.num_layers === 'number') setNumLayers(d.num_layers);
      if (d && d.default_layer != null) setEmbLayer(String(d.default_layer));
    }).catch(() => {});
  }, []);

  const reset = () => { setMeta([]); setAudio(null); setSourceName(''); setResults(null); setError(''); setLoadErr(''); };

  const ingest = (metaText, audioMap, name) => {
    if (!metaText) { setLoadErr('No metadata.csv found in the upload.'); return; }
    const rows = rowsToObjects(parseCSV(metaText));
    if (!rows.length || !('word' in rows[0]) || !('filename' in rows[0])) {
      setLoadErr('metadata.csv must have at least "filename" and "word" columns.'); return;
    }
    setLoadErr(''); setResults(null); setError('');
    setMeta(rows); setAudio(audioMap); setSourceName(name);
  };

  const onZip = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    setLoading(true); reset();
    try {
      const entries = await readZip(await file.arrayBuffer());
      let metaText = '';
      const map = new Map();
      for (const en of entries) {
        if (basename(en.name).toLowerCase() === 'metadata.csv') metaText = new TextDecoder().decode(en.bytes);
        else if (en.name.toLowerCase().endsWith('.wav')) map.set(basename(en.name), new Blob([en.bytes], { type: 'audio/wav' }));
      }
      ingest(metaText, map, file.name);
    } catch (err) {
      setLoadErr('Failed to read ZIP: ' + (err.message || err));
    } finally { setLoading(false); if (zipInput.current) zipInput.current.value = ''; }
  };

  const onDir = async (e) => {
    const files = Array.from(e.target.files || []); if (!files.length) return;
    setLoading(true); reset();
    try {
      let metaText = '';
      const map = new Map();
      for (const f of files) {
        const bn = basename(f.name);
        if (bn.toLowerCase() === 'metadata.csv') metaText = await f.text();
        else if (bn.toLowerCase().endsWith('.wav')) map.set(bn, f);
      }
      const folder = basename(files[0].webkitRelativePath?.split('/')[0] || 'folder');
      ingest(metaText, map, folder);
    } catch (err) {
      setLoadErr('Failed to read folder: ' + (err.message || err));
    } finally { setLoading(false); if (dirInput.current) dirInput.current.value = ''; }
  };

  // Rows that have both a word and a matching audio file
  const scorable = useMemo(() => {
    if (!meta.length || !audio) return [];
    return meta.map((r) => ({ ...r, _bn: basename(r.filename), _hasAudio: audio.has(basename(r.filename)) }));
  }, [meta, audio]);
  const missingAudio = scorable.filter((r) => !r._hasAudio).length;

  const runBatch = async () => {
    setError(''); setResults(null);
    const layerSpec = embLayer.trim();
    if (layerSpec && !/^\d+(\s*,\s*\d+)*$/.test(layerSpec)) { setError('Invalid layer spec — use e.g. 20 or 19,20,21'); return; }
    const rows = scorable.filter((r) => r._hasAudio && (r.word || '').trim());
    if (!rows.length) { setError('Nothing to score (no rows with both audio and a word).'); return; }

    const form = new FormData();
    for (const r of rows) { form.append('files', audio.get(r._bn), r.filename); form.append('words', r.word); }
    if (layerSpec) form.append('layer', layerSpec.replace(/\s+/g, ''));

    setScoring(true);
    try {
      const res = await fetch('/api/batch_score', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      // merge scores back onto metadata rows by filename
      const byName = new Map((data.results || []).map((x) => [x.filename, x]));
      const merged = rows.map((r) => ({ ...r, ...(byName.get(r.filename) || {}) }));
      setResults({ rows: merged, ...data });
    } catch (e) {
      setError(e.message || 'Batch scoring failed');
    } finally { setScoring(false); }
  };

  const passCount = results ? results.rows.filter((r) => r.score != null && r.score >= threshold).length : 0;

  const downloadCsv = () => {
    if (!results) return;
    const cols = ['filename', 'speaker_id', 'word', 'take', 'list_index', 'score', 'pass', 'in_library', 'n_templates', 'error'];
    const lines = [cols.join(',')];
    for (const r of results.rows) {
      const pass = r.score == null ? '' : (r.score >= threshold ? 1 : 0);
      lines.push(cols.map((c) => csvCell(c === 'pass' ? pass : (c === 'score' && r.score != null ? r.score.toFixed(4) : r[c]))).join(','));
    }
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `scores_${sourceName.replace(/\.zip$/i, '')}_thr${threshold.toFixed(2)}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  };

  return (
    <main className="relative min-h-screen bg-gradient-to-b from-slate-50 via-slate-50 to-white dark:from-slate-950 dark:via-slate-950 dark:to-black font-sans bg-grain">
      <div className="sticky top-0 z-40 backdrop-blur bg-white/70 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge className="border border-brand/50 bg-white text-brand dark:bg-transparent"><Waves className="w-3.5 h-3.5 mr-1" /> UW‑Madison</Badge>
            <Badge className="bg-slate-900 text-white dark:bg-slate-200 dark:text-slate-900">Batch Pronunciation Scoring</Badge>
          </div>
          <div className="flex items-center gap-2">
            <PageTabs page={page} onNavigate={onNavigate} />
            <Button variant="subtle" onClick={toggleDark} className="px-3">{dark ? 'Light' : 'Dark'}</Button>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12 space-y-8">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">Batch Score a Session</h1>
          <p className="text-gray-600 dark:text-slate-300 mt-2 max-w-2xl leading-relaxed">Upload a collected session (the ZIP from <b>Collect Data</b>, or its unzipped folder). Each recording is scored against the reference-library prototype for its <code>word</code>, then you get a results table + CSV.</p>
        </div>

        {/* Upload */}
        <Card className="glow">
          <div className="flex flex-wrap items-center gap-3">
            <input ref={zipInput} type="file" accept=".zip" className="hidden" onChange={onZip} />
            <input ref={dirInput} type="file" multiple className="hidden" onChange={onDir} />
            <Button onClick={() => zipInput.current?.click()} disabled={loading}><FileArchive className="w-4 h-4" /> Upload session ZIP</Button>
            <Button variant="outline" onClick={() => dirInput.current?.click()} disabled={loading}><FolderUp className="w-4 h-4" /> Upload folder</Button>
            {loading && <span className="text-sm text-slate-500 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> reading…</span>}
            {sourceName && !loading && <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">{sourceName}: {meta.length} rows</Badge>}
          </div>
          {loadErr && <p className="text-sm text-red-600 dark:text-red-400 mt-3">{loadErr}</p>}
          {meta.length > 0 && (
            <p className="text-xs text-gray-500 dark:text-slate-400 mt-3">
              {scorable.filter((r) => r._hasAudio).length} of {meta.length} rows matched to audio{missingAudio ? ` · ${missingAudio} missing audio` : ''}.
            </p>
          )}
        </Card>

        {/* Settings + run */}
        {meta.length > 0 && (
          <Card className="glow">
            <div className="flex flex-wrap items-end gap-6">
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Embedding layer</div>
                <div className="text-xs text-gray-500 dark:text-slate-400 mb-1">one (20) or fused (19,20,21); empty = final{numLayers ? ` · 1–${numLayers}` : ''}</div>
                <input type="text" inputMode="numeric" value={embLayer} onChange={(e) => setEmbLayer(e.target.value)} placeholder="final"
                  className="w-32 px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-center text-slate-900 dark:text-white tabular-nums outline-none focus:ring-2 focus:ring-brand" />
              </div>
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Pass threshold</div>
                <div className="text-xs text-gray-500 dark:text-slate-400 mb-1">score ≥ threshold → correct</div>
                <div className="flex items-center gap-2">
                  <input type="range" min="0" max="1" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-40 accent-brand" />
                  <span className="text-sm font-semibold tabular-nums text-slate-900 dark:text-white">{threshold.toFixed(2)}</span>
                </div>
              </div>
              <div>
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Results view</div>
                <div className="text-xs text-gray-500 dark:text-slate-400 mb-1">hide the numbers if you only need the verdict</div>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer select-none">
                  <input type="checkbox" checked={simpleView} onChange={(e) => setSimpleView(e.target.checked)} className="accent-brand w-4 h-4" />
                  Correct / Not correct only
                </label>
              </div>
              <div className="flex-1" />
              <Button size="default" onClick={runBatch} disabled={scoring}>
                {scoring ? <Loader2 className="w-4 h-4 animate-spin" /> : <Info className="w-4 h-4" />}
                {scoring ? 'Scoring…' : `Score ${scorable.filter((r) => r._hasAudio && (r.word || '').trim()).length}`}
              </Button>
            </div>
            {scoring && <p className="text-xs text-gray-500 dark:text-slate-400 mt-3">First run builds a prototype per distinct word (cached afterward) — on CPU this can take a while for many words.</p>}
            {error && <p className="text-sm text-red-600 dark:text-red-400 mt-3">{error}</p>}
          </Card>
        )}

        {/* Results */}
        {results && (
          <Card className="glow">
            <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
              <div className="flex items-center gap-3 flex-wrap">
                <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Results</h2>
                <Badge className="bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300">{results.n_scored} scored</Badge>
                {!simpleView && results.mean_score != null && <Badge className="bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300">mean {results.mean_score.toFixed(3)}</Badge>}
                <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">{passCount}/{results.n_scored} {simpleView ? 'correct' : 'pass'}</Badge>
                {results.n_skipped > 0 && <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">{results.n_skipped} skipped</Badge>}
              </div>
              <Button onClick={downloadCsv}><Download className="w-4 h-4" /> Download CSV</Button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-500 dark:text-slate-400 border-b dark:border-slate-800">
                    <th className="py-2 pr-3 font-medium">word</th>
                    <th className="py-2 pr-3 font-medium">speaker</th>
                    <th className="py-2 pr-3 font-medium">take</th>
                    {!simpleView && <th className="py-2 pr-3 font-medium">score</th>}
                    <th className="py-2 pr-3 font-medium">{simpleView ? 'result' : 'pass'}</th>
                    <th className="py-2 pr-3 font-medium">note</th>
                  </tr>
                </thead>
                <tbody>
                  {results.rows.map((r, i) => {
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
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </main>
  );
}
