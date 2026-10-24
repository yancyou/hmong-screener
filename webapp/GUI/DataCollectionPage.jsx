import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Mic, Square, Trash2, Download, Play, RefreshCw, Waves, Loader2, ListChecks, Radio } from 'lucide-react';
import { cn, Card, Button, Badge, Progress, PageTabs } from './ui.jsx';
import { useRecorder, encodeWAV, wavBlob, peakLevel } from './audio.js';
import { makeZip } from './zip.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const safe = (s) => (s || '').trim().replace(/[\\/:*?"<>|]+/g, '').replace(/\s+/g, '_') || 'x';
const nowStamp = () => {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};
const csvCell = (v) => {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export default function DataCollectionPage({ page = 'collect', onNavigate }) {
  const [dark, setDark] = useState(() => (typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false));
  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    try { localStorage.setItem('theme', next ? 'dark' : 'light'); } catch {}
  };
  useEffect(() => {
    try {
      const saved = localStorage.getItem('theme');
      if (saved) { const d = saved === 'dark'; setDark(d); document.documentElement.classList.toggle('dark', d); }
    } catch {}
  }, []);

  // Session metadata
  const [speaker, setSpeaker] = useState('');
  const [gender, setGender] = useState('');
  const [notes, setNotes] = useState('');
  const [mode, setMode] = useState('single'); // 'single' | 'list'

  // Recorder
  const {
    armMic, startCapture, stopCapture, releaseMic, micReady, vu,
    inputGain, setInputGain, devices, deviceId, selectDevice,
  } = useRecorder();
  useEffect(() => () => releaseMic(), [releaseMic]);

  // Recordings: { id, word, listIndex, take, samples, sampleRate, durationSec, timestamp, peak, url }
  const [recordings, setRecordings] = useState([]);
  const idRef = useRef(1);
  useEffect(() => () => { recordings.forEach((r) => URL.revokeObjectURL(r.url)); }, []); // eslint-disable-line

  const takeFor = useCallback((word) => (
    recordings.filter((r) => r.word === word).length + 1
  ), [recordings]);

  const addRecording = useCallback((word, listIndex, samples, sampleRate) => {
    if (!samples || samples.length === 0) return;
    const durationSec = samples.length / sampleRate;
    const url = URL.createObjectURL(wavBlob(samples, sampleRate));
    setRecordings((prev) => {
      const take = prev.filter((r) => r.word === word).length + 1;
      return [...prev, {
        id: idRef.current++, word, listIndex: listIndex == null ? '' : listIndex,
        take, samples, sampleRate, durationSec, timestamp: new Date(), peak: peakLevel(samples), url,
      }];
    });
  }, []);

  const deleteRecording = (id) => setRecordings((prev) => {
    const r = prev.find((x) => x.id === id);
    if (r) URL.revokeObjectURL(r.url);
    return prev.filter((x) => x.id !== id);
  });

  const replaceRecording = (id, samples, sampleRate) => setRecordings((prev) => prev.map((r) => {
    if (r.id !== id) return r;
    URL.revokeObjectURL(r.url);
    return { ...r, samples, sampleRate, durationSec: samples.length / sampleRate, timestamp: new Date(), peak: peakLevel(samples), url: URL.createObjectURL(wavBlob(samples, sampleRate)) };
  }));

  // ---- Single-word mode ----
  const [singleWord, setSingleWord] = useState('');
  const [recordingSingle, setRecordingSingle] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const elapsedTimer = useRef(null);

  const startSingle = async () => {
    if (!singleWord.trim()) return;
    try { await armMic(); } catch { alert('Microphone permission is required'); return; }
    startCapture();
    setRecordingSingle(true);
    setElapsed(0);
    const t0 = performance.now();
    elapsedTimer.current = setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
  };
  const stopSingle = () => {
    clearInterval(elapsedTimer.current);
    const { samples, sampleRate } = stopCapture();
    setRecordingSingle(false);
    addRecording(singleWord.trim(), null, samples, sampleRate);
  };

  // ---- List (guided fixed-window) mode ----
  const [wordsInput, setWordsInput] = useState('');
  // Accept commas *and* newlines, so a word list pasted or loaded from a file
  // (one word per line) works without reformatting.
  const words = useMemo(
    () => wordsInput.split(/[\n,]+/).map((w) => w.trim()).filter(Boolean),
    [wordsInput],
  );

  // Load a word list from a .txt/.csv file. For CSV we take the first column;
  // a leading "word" header row is dropped.
  const loadWordFile = async (file) => {
    if (!file) return;
    try {
      const text = await file.text();
      let items = text.split(/\r?\n/).map((line) => line.split(',')[0].trim()).filter(Boolean);
      if (items.length && /^words?$/i.test(items[0])) items = items.slice(1);
      if (!items.length) { alert('No words found in that file.'); return; }
      setWordsInput(items.join(', '));
    } catch {
      alert('Could not read that file.');
    }
  };
  const [windowSec, setWindowSec] = useState(3);
  const [leadIn, setLeadIn] = useState(3);
  const [gapMs, setGapMs] = useState(600);

  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState('idle'); // idle | countdown | recording | gap | done
  const [curIndex, setCurIndex] = useState(-1);
  const [countdown, setCountdown] = useState(0);
  const [recFrac, setRecFrac] = useState(0); // 0..1 elapsed within window
  const abortRef = useRef(false);
  const recFracTimer = useRef(null);

  const runGuided = async (startFrom = 0) => {
    if (!words.length) return;
    try { await armMic(); } catch { alert('Microphone permission is required'); return; }
    abortRef.current = false;
    setRunning(true);
    for (let i = startFrom; i < words.length; i++) {
      if (abortRef.current) break;
      setCurIndex(i);
      for (let c = leadIn; c > 0; c--) {
        if (abortRef.current) break;
        setPhase('countdown'); setCountdown(c);
        await sleep(1000);
      }
      if (abortRef.current) break;
      setPhase('recording'); setRecFrac(0);
      startCapture();
      const t0 = performance.now();
      recFracTimer.current = setInterval(() => setRecFrac(Math.min(1, (performance.now() - t0) / (windowSec * 1000))), 50);
      await sleep(windowSec * 1000);
      clearInterval(recFracTimer.current);
      const { samples, sampleRate } = stopCapture();
      if (abortRef.current) break;
      addRecording(words[i], i, samples, sampleRate);
      setPhase('gap');
      await sleep(gapMs);
    }
    clearInterval(recFracTimer.current);
    setPhase(abortRef.current ? 'idle' : 'done');
    setCurIndex(-1);
    setRunning(false);
  };
  const abortGuided = () => { abortRef.current = true; clearInterval(recFracTimer.current); };

  // Re-record one row using a single fixed window (with lead-in)
  const [reRecId, setReRecId] = useState(null);
  const reRecord = async (row) => {
    try { await armMic(); } catch { alert('Microphone permission is required'); return; }
    setReRecId(row.id);
    for (let c = leadIn; c > 0; c--) { setPhase('countdown'); setCountdown(c); await sleep(1000); }
    setPhase('recording'); setRecFrac(0);
    startCapture();
    const t0 = performance.now();
    recFracTimer.current = setInterval(() => setRecFrac(Math.min(1, (performance.now() - t0) / (windowSec * 1000))), 50);
    await sleep(windowSec * 1000);
    clearInterval(recFracTimer.current);
    const { samples, sampleRate } = stopCapture();
    replaceRecording(row.id, samples, sampleRate);
    setReRecId(null); setPhase('idle');
  };

  // ---- Export ----
  const [exporting, setExporting] = useState(false);
  const canExport = speaker.trim() && recordings.length > 0;
  const exportZip = async () => {
    if (!canExport) return;
    setExporting(true);
    try {
      const sp = safe(speaker);
      const files = [];
      const meta = [];
      const cols = ['filename', 'speaker_id', 'word', 'list_index', 'take', 'timestamp_iso', 'duration_sec', 'sample_rate', 'gender', 'notes'];
      for (const r of recordings) {
        const fname = `${sp}/${sp}__${safe(r.word)}__take${r.take}.wav`;
        files.push({ name: fname, data: encodeWAV(r.samples, r.sampleRate) });
        meta.push({
          filename: fname, speaker_id: speaker.trim(), word: r.word, list_index: r.listIndex,
          take: r.take, timestamp_iso: r.timestamp.toISOString(), duration_sec: r.durationSec.toFixed(3),
          sample_rate: r.sampleRate, gender, notes,
        });
      }
      const csv = [cols.join(','), ...meta.map((m) => cols.map((c) => csvCell(m[c])).join(','))].join('\n') + '\n';
      files.push({ name: 'metadata.csv', data: csv });
      files.push({ name: 'metadata.json', data: JSON.stringify({ speaker_id: speaker.trim(), gender, notes, exported_at: new Date().toISOString(), recordings: meta }, null, 2) });
      const blob = makeZip(files);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${sp}_${nowStamp()}.zip`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    } finally {
      setExporting(false);
    }
  };

  const clearAll = () => {
    if (!recordings.length || confirm('Delete all recordings in this session?')) {
      recordings.forEach((r) => URL.revokeObjectURL(r.url));
      setRecordings([]);
    }
  };

  const totalDur = recordings.reduce((s, r) => s + r.durationSec, 0);
  const guidedBusy = running || reRecId != null;

  return (
    <main className="relative min-h-screen bg-gradient-to-b from-slate-50 via-slate-50 to-white dark:from-slate-950 dark:via-slate-950 dark:to-black font-sans bg-grain">
      <div className="sticky top-0 z-40 backdrop-blur bg-white/70 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge className="border border-brand/50 bg-white text-brand dark:bg-transparent"><Waves className="w-3.5 h-3.5 mr-1" /> UW‑Madison</Badge>
            <Badge className="bg-slate-900 text-white dark:bg-slate-200 dark:text-slate-900">Hmong Data Collection</Badge>
          </div>
          <div className="flex items-center gap-2">
            <PageTabs page={page} onNavigate={onNavigate} />
            <Button variant="subtle" onClick={toggleDark} className="px-3">{dark ? 'Light' : 'Dark'}</Button>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12 space-y-8">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">Hmong Word Recording</h1>
          <p className="text-gray-600 dark:text-slate-300 mt-2 max-w-2xl leading-relaxed">Record word productions and download the session as a ZIP (audio + metadata). No data leaves this browser until you download.</p>
        </div>

        {/* Session setup */}
        <Card className="glow">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">Session</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-slate-300">Speaker ID <span className="text-brand">*</span></label>
              <input value={speaker} onChange={(e) => setSpeaker(e.target.value)} placeholder="e.g. YX001"
                className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand" />
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-slate-300">Gender / group</label>
              <input value={gender} onChange={(e) => setGender(e.target.value)} placeholder="optional"
                className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand" />
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-slate-300">Notes</label>
              <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="optional"
                className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand" />
            </div>
          </div>
          {!speaker.trim() && <p className="text-xs text-amber-600 dark:text-amber-400 mt-3">Enter a Speaker ID before exporting.</p>}
        </Card>

        {/* Microphone: pick the device, set the level, watch the meter */}
        <Card className="glow">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Microphone</div>
            {!micReady && (
              <Button variant="subtle" onClick={() => armMic().catch(() => alert('Microphone permission is required'))}>
                <Mic className="w-4 h-4" /> Connect microphone
              </Button>
            )}
          </div>

          {micReady ? (
            <div className="mt-3 space-y-3">
              {devices.length > 1 && (
                <div>
                  <label className="text-xs text-gray-500 dark:text-slate-400">Input device</label>
                  <select value={deviceId} disabled={guidedBusy || recordingSingle}
                    onChange={(e) => selectDevice(e.target.value)}
                    className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand">
                    {devices.map((d, i) => (
                      <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Microphone ${i + 1}`}</option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-500 dark:text-slate-400">Input volume</label>
                  <span className="text-xs font-semibold tabular-nums text-slate-700 dark:text-slate-300">{inputGain.toFixed(1)}×</span>
                </div>
                <div className="flex items-center gap-3">
                  <input type="range" min="0.1" max="5" step="0.1" value={inputGain}
                    onChange={(e) => setInputGain(Number(e.target.value))}
                    className="flex-1 accent-brand" />
                  <Button variant="subtle" onClick={() => setInputGain(1)} title="Reset to 1.0×">Reset</Button>
                </div>
              </div>

              <div>
                <div className="text-xs text-gray-500 dark:text-slate-400 mb-1">
                  Level — speak at a normal volume; aim for the green band
                </div>
                <div className="relative h-3 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  {/* target band: comfortably loud but not clipping */}
                  <div className="absolute inset-y-0 bg-emerald-500/20" style={{ left: '35%', width: '45%' }} />
                  <div className={cn('h-full transition-all',
                    vu > 0.92 ? 'bg-red-500' : vu > 0.35 ? 'bg-emerald-500' : 'bg-amber-400')}
                    style={{ width: `${Math.round(Math.min(1, vu) * 100)}%` }} />
                </div>
                <div className="text-xs mt-1 h-4">
                  {vu > 0.92
                    ? <span className="text-red-600 dark:text-red-400">Too loud — turn the input volume down</span>
                    : vu > 0.35
                      ? <span className="text-emerald-600 dark:text-emerald-400">Good level</span>
                      : vu > 0.02
                        ? <span className="text-amber-600 dark:text-amber-400">Quiet — turn the input volume up or move closer</span>
                        : <span className="text-slate-400">Silent</span>}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-gray-500 dark:text-slate-400 mt-2">
              Connect the microphone to check the recording level before you start.
            </p>
          )}
        </Card>

        {/* Mode switch */}
        <div className="inline-flex rounded-2xl bg-slate-100 dark:bg-slate-800 p-1">
          <button onClick={() => setMode('single')} disabled={guidedBusy}
            className={cn('px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2 transition', mode === 'single' ? 'bg-white text-brand shadow dark:bg-slate-900' : 'text-slate-500 hover:text-slate-800 dark:text-slate-400')}>
            <Radio className="w-4 h-4" /> Single word
          </button>
          <button onClick={() => setMode('list')} disabled={guidedBusy}
            className={cn('px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2 transition', mode === 'list' ? 'bg-white text-brand shadow dark:bg-slate-900' : 'text-slate-500 hover:text-slate-800 dark:text-slate-400')}>
            <ListChecks className="w-4 h-4" /> Word list
          </button>
        </div>

        {/* Single-word mode */}
        {mode === 'single' && (
          <Card className="glow">
            <div className="flex flex-col sm:flex-row sm:items-end gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium text-gray-700 dark:text-slate-300">Word</label>
                <input value={singleWord} onChange={(e) => setSingleWord(e.target.value)} placeholder="type the word being recorded"
                  disabled={recordingSingle}
                  className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand" />
              </div>
              {!recordingSingle ? (
                <Button size="lg" onClick={startSingle} disabled={!singleWord.trim()}><Mic className="w-5 h-5" /> Record</Button>
              ) : (
                <Button size="lg" variant="danger" onClick={stopSingle}><Square className="w-5 h-5" /> Stop {elapsed ? `${elapsed.toFixed(1)}s` : ''}</Button>
              )}
            </div>
            <div className="mt-4 flex items-center gap-3">
              <div className="flex-1 h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                <div className="h-full bg-brand transition-all" style={{ width: `${Math.round(vu * 100)}%` }} />
              </div>
              {recordingSingle && <span className="text-xs text-red-600 dark:text-red-400 font-medium animate-pulse">● recording</span>}
            </div>
            <p className="text-xs text-gray-500 dark:text-slate-400 mt-3">Press Record, say the word, press Stop. Repeat for more takes — each take is saved separately.</p>
          </Card>
        )}

        {/* Word-list guided mode */}
        {mode === 'list' && (
          <Card className="glow">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <label className="text-sm font-medium text-gray-700 dark:text-slate-300">Words (one per line, or comma-separated)</label>
              <label className={cn(
                'text-xs px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 cursor-pointer',
                'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300',
                guidedBusy && 'opacity-50 pointer-events-none',
              )}>
                Load word list (.txt / .csv)
                <input type="file" accept=".txt,.csv,text/plain,text/csv" className="hidden"
                  disabled={guidedBusy}
                  onChange={(e) => { loadWordFile(e.target.files?.[0]); e.target.value = ''; }} />
              </label>
            </div>
            <textarea value={wordsInput} onChange={(e) => setWordsInput(e.target.value)} disabled={guidedBusy} rows={3}
              placeholder={'dhia, dhiab, dhias, dhiaj, …\nor one word per line'}
              className="mt-1 w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-brand" />
            <div className="mt-3 flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs text-gray-500 dark:text-slate-400">Window (s)</label>
                <input type="number" min="1" max="15" step="0.5" value={windowSec} disabled={guidedBusy}
                  onChange={(e) => setWindowSec(Math.max(0.5, Number(e.target.value) || 3))}
                  className="mt-1 w-20 px-2 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-center text-slate-900 dark:text-white tabular-nums" />
              </div>
              <div>
                <label className="text-xs text-gray-500 dark:text-slate-400">Lead-in (s)</label>
                <input type="number" min="0" max="10" step="1" value={leadIn} disabled={guidedBusy}
                  onChange={(e) => setLeadIn(Math.max(0, Math.round(Number(e.target.value) || 0)))}
                  className="mt-1 w-20 px-2 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-center text-slate-900 dark:text-white tabular-nums" />
              </div>
              <div>
                <label className="text-xs text-gray-500 dark:text-slate-400">Gap (ms)</label>
                <input type="number" min="0" max="3000" step="100" value={gapMs} disabled={guidedBusy}
                  onChange={(e) => setGapMs(Math.max(0, Number(e.target.value) || 0))}
                  className="mt-1 w-24 px-2 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-center text-slate-900 dark:text-white tabular-nums" />
              </div>
              <div className="flex-1" />
              {!running ? (
                <Button size="lg" onClick={() => runGuided(0)} disabled={!words.length || reRecId != null}><Mic className="w-5 h-5" /> Start ({words.length})</Button>
              ) : (
                <Button size="lg" variant="danger" onClick={abortGuided}><Square className="w-5 h-5" /> Stop</Button>
              )}
            </div>

            {/* Live guided panel */}
            {(running || reRecId != null) && (
              <div className="mt-5 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 text-center bg-slate-50 dark:bg-slate-900">
                <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">
                  {running ? `Word ${curIndex + 1} / ${words.length}` : 're-recording'}
                </div>
                <div className="text-4xl font-extrabold text-slate-900 dark:text-white mb-3">
                  {running ? words[curIndex] : recordings.find((r) => r.id === reRecId)?.word}
                </div>
                {phase === 'countdown' && <div className="text-6xl font-black text-brand tabular-nums">{countdown}</div>}
                {phase === 'recording' && (
                  <div className="space-y-2">
                    <div className="text-sm text-red-600 dark:text-red-400 font-semibold animate-pulse">● RECORDING — say the word now</div>
                    <Progress value={(1 - recFrac) * 100} />
                  </div>
                )}
                {phase === 'gap' && <div className="text-sm text-slate-400">saved ✓</div>}
                <div className="mt-4 h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden max-w-xs mx-auto">
                  <div className="h-full bg-brand transition-all" style={{ width: `${Math.round(vu * 100)}%` }} />
                </div>
              </div>
            )}
            {!running && reRecId == null && (
              <p className="text-xs text-gray-500 dark:text-slate-400 mt-3">Each word shows a lead-in countdown, then records for the fixed window and auto-advances. Re-record any word afterward from the list below.</p>
            )}
          </Card>
        )}

        {/* Collected recordings */}
        <Card className="glow">
          <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Recordings <span className="text-slate-400 font-normal">({recordings.length}{recordings.length ? `, ${totalDur.toFixed(1)}s` : ''})</span></h2>
            <div className="flex items-center gap-2">
              {recordings.length > 0 && <Button variant="ghost" onClick={clearAll}><Trash2 className="w-4 h-4" /> Clear</Button>}
              <Button onClick={exportZip} disabled={!canExport || exporting}>
                {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />} Download ZIP
              </Button>
            </div>
          </div>
          {recordings.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-slate-400">No recordings yet.</p>
          ) : (
            <div className="space-y-2">
              {recordings.map((r) => (
                <div key={r.id} className="flex items-center gap-3 rounded-xl border dark:border-slate-800 p-2.5 bg-slate-50 dark:bg-slate-900">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">{r.word}</span>
                      <Badge className="bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300">take {r.take}</Badge>
                      {r.listIndex !== '' && <Badge className="bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300">#{r.listIndex + 1}</Badge>}
                      {r.peak < 0.02 && <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">low level</Badge>}
                    </div>
                    <div className="text-xs text-slate-400">{r.durationSec.toFixed(2)}s · {r.sampleRate} Hz</div>
                  </div>
                  <audio controls src={r.url} className="h-9 max-w-[200px]" />
                  <Button variant="ghost" size="icon" title="Re-record (fixed window)" disabled={guidedBusy} onClick={() => reRecord(r)}><RefreshCw className="w-4 h-4" /></Button>
                  <Button variant="ghost" size="icon" title="Delete" disabled={guidedBusy} onClick={() => deleteRecording(r.id)}><Trash2 className="w-4 h-4" /></Button>
                </div>
              ))}
            </div>
          )}
          {canExport && <p className="text-xs text-gray-500 dark:text-slate-400 mt-4">ZIP contains WAV files under <code>{safe(speaker)}/</code> plus <code>metadata.csv</code> and <code>metadata.json</code>.</p>}
        </Card>
      </div>
    </main>
  );
}
