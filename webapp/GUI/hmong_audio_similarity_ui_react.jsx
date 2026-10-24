import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Upload, Mic, Play, Pause, Trash2, Waves, Headphones, Info, Loader2, Pencil, Download } from "lucide-react";
import { PageTabs } from "./ui.jsx";

// --- Minimal shadcn-style components (inline) ---
const cn = (...c) => c.filter(Boolean).join(" ");
const Card = ({ className = "", children }) => (
  <div className={cn("rounded-3xl shadow-xl border bg-white p-6 transition-all duration-300 hover:shadow-2xl dark:bg-slate-900 dark:border-slate-800", className)}>{children}</div>
);
const Button = ({ className = "", variant = "default", size = "default", disabled, children, ...props }) => {
  const base = "relative overflow-hidden rounded-2xl transform-gpu will-change-transform transition-all duration-300 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed focus:ring-brand dark:focus:ring-offset-slate-900 hover:-translate-y-1 hover:scale-105 hover:shadow-[0_14px_32px_-8px_rgba(197,5,12,0.55)] active:translate-y-0 active:scale-100";
  const sizes = {
    default: "px-5 py-2.5 text-sm font-semibold gap-2",
    icon: "p-2.5",
  };
  const variants = {
    default: "bg-brand text-white hover:bg-brand/90",
    ghost: "bg-transparent border border-slate-300 text-slate-700 hover:bg-slate-100 hover:border-slate-400 hover:text-slate-900 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800/80 dark:hover:border-slate-500",
    outline: "bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-slate-400 dark:bg-slate-900 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800 dark:hover:border-slate-500",
    subtle: "bg-slate-100 text-slate-800 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700",
  };
  return (
    <button disabled={disabled} className={cn(base, sizes[size], variants[variant], className)} {...props}>
      {children}
    </button>
  );
};
const Label = ({ children, className = "" }) => (
  <label className={cn("text-sm font-medium text-gray-700 dark:text-slate-300", className)}>{children}</label>
);

// i18n
const TEXT = {
  EN: {
    uwBadge: "UW‑Madison",
    modelBadge: "Hmong XLS‑R",
    title: "Cross‑lingual Audio Similarity",
    desc: "Upload a reference clip and a candidate clip.",
    reference: "Reference Audio",
    candidate: "Candidate Audio",
    chooseFile: "Choose a WAV file",
    tip: "Tip: keep clips 2–10 seconds for best latency.",
    reset: "Reset",
    compute: "Compute Score",
    scoring: "Scoring…",
    decision: "Decision threshold",
    decisionDesc: "Predictions ≥ threshold are considered similar.",
    layerTitle: "Embedding layer",
    layerDesc: "Layer(s) for the embedding — one layer (e.g. 20) or fused layers (e.g. 19,20,21). Empty = final output. Multiple reference clips are averaged into a template prototype.",
    layerFinal: "final",
    wordPlaceholder: "Type to search the word library…",
    wordTemplates: "library templates",
    wordClear: "Clear",
    wordUnknown: "Word not in library",
    orUpload: "…or upload / record your own below (added to the templates)",
    similarity: "Similarity",
    pass: "Pass",
    below: "Below threshold",
    selectTwo: "Select two clips and click Compute Score to see results.",
    route: "Expected backend route",
    uploading: "Uploading…",
    computed: "Computed",
    pleaseBoth: "Please select both files",
    onlyWav: "Please select a WAV file",
    refLoaded: "Reference loaded",
    candLoaded: "Candidate loaded",
    dark: "Dark",
    light: "Light",
    ref1s: "A · Ref 1s",
    cand1s: "B · Cand 1s",
    record: "Record",
    stop: "Stop",
    recordNote: "Recording is saved as WAV",
    renamePrompt: "Enter new name (extension optional)",
  },
  HMN: {
    uwBadge: "UW‑Madison",
    modelBadge: "Hmong XLS‑R",
    title: "Kev sib xws ntawm suab lus",
    desc: "Upload ib daim suab qhia thiab ib daim suab xeem.",
    reference: "Suab Qhia",
    candidate: "Suab Xeem",
    chooseFile: "Xaiv ib daim suab WAV",
    tip: "Tswv yim: khaws 2–10 vib nas this kom qeeb qis.",
    reset: "Rov pib",
    compute: "Xam qhab nia",
    scoring: "Tab xam…",
    decision: "Qhov rooj vag txiav txim",
    decisionDesc: "Yog qhab nia ≥ rooj vag ces suav tias sib xws.",
    layerTitle: "Txheej embedding",
    layerDesc: "Xaiv ib txheej (20) los ntau txheej ua ke (19,20,21). Tso seem yog siv qhov kawg. Ntau daim suab qhia yuav muab ua ib qho nruab nrab.",
    layerFinal: "kawg",
    wordPlaceholder: "Ntaus nrhiav lo lus hauv lub tsev khaws…",
    wordTemplates: "daim qauv hauv tsev khaws",
    wordClear: "Tshem",
    wordUnknown: "Tsis muaj lo lus no",
    orUpload: "…los sis upload / kaw koj tus kheej (ntxiv rau cov qauv)",
    similarity: "Qib sib xws",
    pass: "Hla",
    below: "Tsis txog",
    selectTwo: "Xaiv ob daim suab thiab nias “Xam qhab nia”.",
    route: "Txoj kev tom qab",
    uploading: "Tab tom xa…",
    computed: "Xam tau lawm",
    pleaseBoth: "Thov xaiv ob daim suab",
    onlyWav: "Thov xaiv ntaub suab WAV",
    refLoaded: "Suab qhia tau nqa",
    candLoaded: "Suab xeem tau nqa",
    dark: "Hmo ntuj",
    light: "Kaj ntug",
    ref1s: "A · Qhia 1s",
    cand1s: "B · Xeem 1s",
    record: "Sau suab",
    stop: "Tso tseg",
    recordNote: "Lus ntes yuav khaws ua WAV",
    renamePrompt: "Sau lub npe tshiab (txoj txuas ntxiv .wav yuav ntxiv yog tsis muaj)",
    prepare: "Npaj mic",
    preparing: "Tab npaj…",
    prepared: "Npaj tiav",
  }
};

// Toasts
function useToasts() {
  const [toasts, setToasts] = useState([]);
  const push = useCallback((msg, type = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2500);
  }, []);
  const Toasts = () => (
    <div className="fixed bottom-4 right-4 space-y-2 z-50">
      {toasts.map((t) => (
        <div key={t.id} className={cn("px-3 py-2 rounded-lg shadow-lg text-sm text-white", t.type === "error" ? "bg-red-600" : t.type === "success" ? "bg-emerald-600" : "bg-slate-900/90")}>{t.msg}</div>
      ))}
    </div>
  );
  return { push, Toasts };
}

// Drag & drop + styled FileInput (WAV only)
const FileInput = ({ id, onChange, disabled, placeholder }) => {
  const [fileName, setFileName] = useState("");
  const [drag, setDrag] = useState(false);
  const ref = useRef(null);
  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type !== 'audio/wav' && !file.name.toLowerCase().endsWith('.wav')) {
      // external toast via custom event
      window.dispatchEvent(new CustomEvent('wav-error'));
      if (ref.current) ref.current.value = '';
      setFileName("");
      return;
    }
    setFileName(file ? file.name : "");
    if (onChange) onChange(e);
  };
  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDrag(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (file.type !== 'audio/wav' && !file.name.toLowerCase().endsWith('.wav')) {
        window.dispatchEvent(new CustomEvent('wav-error'));
        return;
      }
      const dt = new DataTransfer();
      dt.items.add(file);
      if (ref.current) ref.current.files = dt.files;
      setFileName(file.name);
      onChange && onChange({ target: { files: dt.files } });
    }
  };
  return (
    <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop} className={cn("relative", disabled && "opacity-60")}>      
      <label htmlFor={id} className={cn("w-full cursor-pointer px-4 py-2 rounded-xl text-sm font-medium transition flex items-center gap-2 bg-white border border-gray-300 hover:bg-gray-50 dark:bg-slate-900 dark:border-slate-700 dark:hover:bg-slate-800", drag && "ring-2 ring-brand ring-offset-2")}>        
        <Upload className="w-4 h-4 text-gray-600 dark:text-slate-300" />
        <span className="truncate text-gray-700 dark:text-slate-200">{fileName || placeholder}</span>
      </label>
      <input ref={ref} id={id} type="file" accept="audio/wav" onChange={handleChange} disabled={disabled} className="sr-only" />
      {drag && <div className="absolute inset-0 rounded-xl border-2 border-dashed border-brand pointer-events-none" />}
    </div>
  );
}

const Progress = ({ value }) => (
  <div className="w-full h-2 bg-gray-200 dark:bg-slate-800 rounded-full overflow-hidden">
    <div className="h-full bg-brand transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
  </div>
);
const Badge = ({ children, className = "" }) => (
  <span className={cn("inline-flex items-center rounded-full px-3 py-1 text-xs font-medium", className)}>{children}</span>
);

// Score is the raw cosine similarity in [-1, 1]
const fmtCos = (x) => x.toFixed(3);

export default function AudioSimilarityPage({ page = 'eval', onNavigate } = {}) {
  const [segments, setSegments] = useState([]); // multi-segment candidate WAVs
  const [refSegments, setRefSegments] = useState([]); // multi-segment reference WAVs
  const [score, setScore] = useState(null);
  const [shownScore, setShownScore] = useState(null); // animated display
  const [loading, setLoading] = useState(false);
  const [explain, setExplain] = useState(null);
  const [error, setError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [threshold, setThreshold] = useState(0.70);
  const [embLayer, setEmbLayer] = useState(""); // 1-based transformer layer; empty = final output
  const [numLayers, setNumLayers] = useState(null);
  const [refWord, setRefWord] = useState(""); // library word used as reference templates
  const [words, setWords] = useState([]); // [{word, n_templates}]
  const [dark, setDark] = useState(() => (typeof window !== 'undefined' ? document.documentElement.classList.contains('dark') : false));
  const [lang, setLang] = useState('EN'); // EN | HMN
  const { push, Toasts } = useToasts();
  const t = TEXT[lang];
  const [micReady, setMicReady] = useState(false);
  const [micWarming, setMicWarming] = useState(false);
  const labelPrepare = t.prepare || 'Prepare mic';
  const labelPreparing = t.preparing || 'Preparing…';
  const labelPrepared = t.prepared || 'Ready';

  // ---- Rename helpers (minimal additive) ----
  const normalizeWavName = (name) => {
    let n = (name || "").trim();
    if (!n) n = `segment_${Date.now()}`;
    n = n.replace(/[\\/:*?"<>|]+/g, "").replace(/\s+/g, "_");
    if (!n.toLowerCase().endsWith('.wav')) n += '.wav';
    return n;
  };
  const handleRenameSegment = (index) => {
    const old = segments[index];
    if (!old) return;
    const input = typeof window !== 'undefined' ? window.prompt(t.renamePrompt, old.name || "segment.wav") : null;
    if (!input) return;
    const newFile = new File([old], normalizeWavName(input), { type: old.type || 'audio/wav' });
    setSegments((prev) => prev.map((f, i) => (i === index ? newFile : f)));
    push('Renamed', 'success');
  };
  const handleRenameRefSegment = (index) => {
    const old = refSegments[index];
    if (!old) return;
    const input = typeof window !== 'undefined' ? window.prompt(t.renamePrompt, old.name || "reference.wav") : null;
    if (!input) return;
    const newFile = new File([old], normalizeWavName(input), { type: old.type || 'audio/wav' });
    setRefSegments((prev) => prev.map((f, i) => (i === index ? newFile : f)));
    push('Renamed', 'success');
  };
  const commitRenameSegment = (index, newName) => {
    const old = segments[index];
    if (!old) return;
    const newFile = new File([old], normalizeWavName(newName), { type: old.type || 'audio/wav' });
    setSegments((prev) => prev.map((f, i) => (i === index ? newFile : f)));
    push('Renamed', 'success');
  };
  const commitRenameRefSegment = (index, newName) => {
    const old = refSegments[index];
    if (!old) return;
    const newFile = new File([old], normalizeWavName(newName), { type: old.type || 'audio/wav' });
    setRefSegments((prev) => prev.map((f, i) => (i === index ? newFile : f)));
    push('Renamed', 'success');
  };

  // Inline editable name component
  function EditableName({ name, onCommit, className = "" }) {
    const [editing, setEditing] = useState(false);
    const [value, setValue] = useState(name);
    useEffect(() => { setValue(name); }, [name]);
    const onKeyDown = (e) => {
      if (e.key === 'Enter') { onCommit(value); setEditing(false); }
      if (e.key === 'Escape') { setValue(name); setEditing(false); }
    };
    const onBlur = () => { onCommit(value); setEditing(false); };
    if (!editing) {
      return (
        <button className={cn("text-sm font-medium truncate text-slate-700 dark:text-slate-200 text-left", className)} title="Click to rename" onClick={() => setEditing(true)}>
          {name}
        </button>
      );
    }
    return (
      <input
        className={cn("text-sm font-medium truncate bg-transparent border-b border-slate-300 dark:border-slate-600 outline-none px-0 text-slate-800 dark:text-slate-100", className)}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        onBlur={onBlur}
        autoFocus
      />
    );
  }

  // Simple icon link styled as a button
  const IconButtonLink = ({ href, filename, className, children }) => {
    const base = "relative overflow-hidden rounded-2xl transform-gpu will-change-transform transition-all duration-300 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand dark:focus:ring-offset-slate-900 hover:-translate-y-1 hover:scale-105 hover:shadow-[0_14px_32px_-8px_rgba(197,5,12,0.55)] active:translate-y-0 active:scale-100";
    const size = "p-2.5";
    const variant = "bg-transparent border border-slate-300 text-slate-700 hover:bg-slate-100 hover:border-slate-400 hover:text-slate-900 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800/80 dark:hover:border-slate-500";
    return (
      <a href={href} download={filename} className={cn(base, size, variant, className)}>
        {children}
      </a>
    );
  };

  // WAV-only error bridge
  useEffect(() => {
    const onErr = () => push(t.onlyWav, 'error');
    window.addEventListener('wav-error', onErr);
    return () => window.removeEventListener('wav-error', onErr);
  }, [push, t.onlyWav]);

  // Audio preview URLs
  const segmentUrls = useMemo(() => segments.map(f => URL.createObjectURL(f)), [segments]);
  const refSegmentUrls = useMemo(() => refSegments.map(f => URL.createObjectURL(f)), [refSegments]);
  useEffect(() => () => { segmentUrls.forEach(u => URL.revokeObjectURL(u)); }, [segmentUrls]);
  useEffect(() => () => { refSegmentUrls.forEach(u => URL.revokeObjectURL(u)); }, [refSegmentUrls]);

  // Waveform drawing
  const refWaveRef = useRef(null);
  const candWaveRef = useRef(null);
  const drawWaveformFromFile = useCallback(async (file, canvas) => {
    if (!file || !canvas) return;
    try {
      const arrayBuffer = await file.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC();
      let buffer;
      try { buffer = await ctx.decodeAudioData(arrayBuffer); }
      catch (err) { buffer = await new Promise((resolve, reject) => ctx.decodeAudioData(arrayBuffer.slice(0), resolve, reject)); }
      const data = buffer.getChannelData(0);
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      canvas.width = width; canvas.height = height;
      const g = canvas.getContext('2d');
      g.clearRect(0, 0, width, height);
      const mid = height / 2; const amp = height * 0.45;
      const step = Math.max(1, Math.floor(data.length / width));
      g.lineWidth = Math.max(1, 0.8 * dpr);
      const grad = g.createLinearGradient(0, 0, width, 0);
      if (dark) { grad.addColorStop(0, 'rgba(255,255,255,0.85)'); grad.addColorStop(1, 'rgba(197,5,12,0.9)'); }
      else { grad.addColorStop(0, 'rgba(30,41,59,0.9)'); grad.addColorStop(1, 'rgba(197,5,12,0.9)'); }
      g.strokeStyle = grad;
      g.beginPath();
      for (let x = 0; x < width; x++) {
        const idx = x * step;
        let min = 1, max = -1;
        for (let j = 0; j < step && idx + j < data.length; j++) {
          const v = data[idx + j];
          if (v < min) min = v; if (v > max) max = v;
        }
        const y1 = mid + min * amp; const y2 = mid + max * amp;
        g.moveTo(x + 0.5, y1); g.lineTo(x + 0.5, y2);
      }
      g.stroke();
      await ctx.close();
    } catch {}
  }, [dark]);

  // Reusable waveform canvas component
  function WaveCanvas({ file }) {
    const canvasRef = useRef(null);
    useEffect(() => { if (file && canvasRef.current) drawWaveformFromFile(file, canvasRef.current); }, [file, drawWaveformFromFile]);
    return <div className="wave-wrap glow mt-2"><canvas ref={canvasRef} /></div>;
  }

  useEffect(() => { if (refSegments.length > 0 && refWaveRef.current) drawWaveformFromFile(refSegments[0], refWaveRef.current); }, [refSegments, drawWaveformFromFile]);
  useEffect(() => { if (segments.length > 0 && candWaveRef.current) drawWaveformFromFile(segments[0], candWaveRef.current); }, [segments, drawWaveformFromFile]);

  // Animated number for score
  useEffect(() => {
    if (score == null) { setShownScore(null); return; }
    let raf; let start; const from = 0; const to = score; const duration = 1100;
    const tick = (ts) => {
      if (!start) start = ts;
      const p = Math.min(1, (ts - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setShownScore(from + (to - from) * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [score]);

  // Recording to WAV with VU
  const mediaRec = useRef(null); // legacy ref not used for WAV now
  const chunksRef = useRef([]); // legacy
  const [recState, setRecState] = useState("idle");
  const [recTime, setRecTime] = useState(0);
  const timerRef = useRef(null);
  const audioCtxRef = useRef(null); // VU context
  const analyserRef = useRef(null);
  const vuRafRef = useRef(null);
  const [vu, setVu] = useState(0);
  const recCtxRef = useRef(null); // recording context
  const procRef = useRef(null);
  const gainRef = useRef(null);
  const recStreamRef = useRef(null);
  const wavBuffersRef = useRef([]);
  const [recCountdown, setRecCountdown] = useState(0);
  const armTimerRef = useRef(null);
  const armedRef = useRef(false);
  const dropSamplesRemainingRef = useRef(0);

  const startTimer = () => {
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setRecTime((t) => t + 1), 1000);
  };
  const stopTimer = () => {
    clearInterval(timerRef.current);
    setRecTime(0);
  };

  const startVU = (stream) => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const loop = () => {
        analyser.getByteFrequencyData(data);
        let avg = 0; for (let i = 0; i < data.length; i++) avg += data[i];
        setVu(avg / data.length / 255);
        vuRafRef.current = requestAnimationFrame(loop);
      };
      loop();
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
    } catch {}
  };
  const stopVU = () => {
    if (vuRafRef.current) cancelAnimationFrame(vuRafRef.current);
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    analyserRef.current = null; setVu(0);
  };

  const encodeWAV = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
    let offset = 0;
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM
    view.setUint16(20, 1, true); // linear PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    // PCM data
    let idx = 44;
    for (let i = 0; i < samples.length; i++, idx += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  };

  const startRecording = async () => {
    try {
      setRecState("arming");
      setRecCountdown(3);
      // begin mic & graphs
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recStreamRef.current = stream;
      // WAV capture graph
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      wavBuffersRef.current = [];
      armedRef.current = false;
      dropSamplesRemainingRef.current = 0;
      processor.onaudioprocess = (e) => {
        if (!armedRef.current) return; // not yet started capturing
        const ch0 = e.inputBuffer.getChannelData(0);
        let startIdx = 0;
        let toDrop = dropSamplesRemainingRef.current || 0;
        if (toDrop > 0) {
          if (toDrop >= ch0.length) {
            dropSamplesRemainingRef.current = toDrop - ch0.length;
            return;
          } else {
            startIdx = toDrop;
            dropSamplesRemainingRef.current = 0;
          }
        }
        const frame = new Float32Array(ch0.length - startIdx);
        frame.set(ch0.subarray(startIdx));
        wavBuffersRef.current.push(frame);
      };
      const gain = ctx.createGain();
      gain.gain.value = 0;
      source.connect(processor);
      processor.connect(gain);
      gain.connect(ctx.destination);
      recCtxRef.current = ctx; procRef.current = processor; gainRef.current = gain;

      // VU
      startVU(stream);

      // countdown to start
      let count = 3;
      if (armTimerRef.current) clearInterval(armTimerRef.current);
      armTimerRef.current = setInterval(() => {
        count -= 1;
        setRecCountdown(count);
        if (count <= 0) {
          clearInterval(armTimerRef.current);
          armTimerRef.current = null;
          // arm now
          armedRef.current = true;
          try { dropSamplesRemainingRef.current = Math.floor((recCtxRef.current?.sampleRate || 44100) * 0.1); } catch { dropSamplesRemainingRef.current = 4410; }
      setRecState("recording");
      startTimer();
        }
      }, 1000);
    } catch (e) {
      setRecState("idle");
      setRecCountdown(0);
      push('Mic permission denied', 'error');
    }
  };

  const stopRecording = () => {
    // stop capture graph
    try {
      if (procRef.current) { procRef.current.disconnect(); procRef.current.onaudioprocess = null; procRef.current = null; }
      if (gainRef.current) { gainRef.current.disconnect(); gainRef.current = null; }
      if (recCtxRef.current) { recCtxRef.current.close(); }
    } catch {}

    // clear arming if any
    if (armTimerRef.current) { clearInterval(armTimerRef.current); armTimerRef.current = null; }
    armedRef.current = false;
    setRecCountdown(0);

    // stop mic
    try { recStreamRef.current?.getTracks().forEach((t) => t.stop()); } catch {}

    // build WAV
    try {
      const buffers = wavBuffersRef.current || [];
      let total = 0; for (const b of buffers) total += b.length;
      const samples = new Float32Array(total);
      let off = 0; for (const b of buffers) { samples.set(b, off); off += b.length; }
      const sr = recCtxRef.current?.sampleRate || 44100;
      const blob = encodeWAV(samples, sr);
      const file = new File([blob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
      setSegments((prev) => [...prev, file]); // append as a new segment
    } catch {}

    recCtxRef.current = null; wavBuffersRef.current = [];

    // stop VU
    stopVU();

      setRecState("idle");
      stopTimer();
  };

  // Mic warm-up: quick permission + device spin-up
  const warmUpMic = async () => {
    if (micReady || micWarming || recState !== 'idle') return;
    setMicWarming(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 44100, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      // quickly stop to free device
      try { stream.getTracks().forEach(t => t.stop()); } catch {}
      setMicReady(true);
      push(t.prepared, 'success');
    } catch (e) {
      push(e.message || 'Mic warm-up failed', 'error');
    } finally {
      setMicWarming(false);
    }
  };

  const resetAll = () => {
    setSegments([]);
    setRefSegments([]);
    setRefWord("");
    setScore(null);
    setExplain(null);
    setError("");
    setUploadProgress(0);
    setMicReady(false);
    setMicWarming(false);
    document.querySelectorAll('input[type="file"]').forEach(input => input.value = '');
    push(lang === 'EN' ? 'Reset done' : 'Rov pib dua');
  };

  // Combine multiple WAV segments into one WAV (reused)
  const resampleFloat32 = (input, inputRate, targetRate) => {
    if (inputRate === targetRate) return input;
    const ratio = inputRate / targetRate;
    const newLen = Math.round(input.length / ratio);
    const output = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = input[idx] || 0;
      const s1 = input[idx + 1] || 0;
      output[i] = s0 + (s1 - s0) * frac;
    }
    return output;
  };

  const decodeWavToPcm = async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    let buffer;
    try { buffer = await ctx.decodeAudioData(arrayBuffer); }
    catch (err) {
      buffer = await new Promise((resolve, reject) => ctx.decodeAudioData(arrayBuffer.slice(0), resolve, reject));
    }
    const data = buffer.getChannelData(0);
    const sr = buffer.sampleRate;
    await ctx.close();
    return { data: new Float32Array(data), sampleRate: sr };
  };

  const combineSegmentsWav = async (files) => {
    if (!files || files.length === 0) return null;
    const decoded = [];
    for (const f of files) {
      // only accept wav per UI, but still safe-guard
      const { data, sampleRate } = await decodeWavToPcm(f);
      decoded.push({ data, sampleRate });
    }
    const targetRate = 44100;
    const resampled = decoded.map(({ data, sampleRate }) => resampleFloat32(data, sampleRate, targetRate));
    let total = 0; for (const d of resampled) total += d.length;
    const merged = new Float32Array(total);
    let off = 0; for (const d of resampled) { merged.set(d, off); off += d.length; }
    const wavBlob = encodeWAV(merged, targetRate);
    return new File([wavBlob], `candidate_${Date.now()}.wav`, { type: 'audio/wav' });
  };

  // Combined preview URLs
  const [combinedUrl, setCombinedUrl] = useState(null);
  const [combinedFile, setCombinedFile] = useState(null);
  const [combiningCandidate, setCombiningCandidate] = useState(false);
  useEffect(() => {
    let cancelled = false;
    if (!segments.length) { if (combinedUrl) URL.revokeObjectURL(combinedUrl); setCombinedUrl(null); setCombinedFile(null); return; }
    setCombiningCandidate(true);
    const handle = setTimeout(async () => {
      try {
        const combined = await combineSegmentsWav(segments);
        if (!cancelled && combined) {
          if (combinedUrl) URL.revokeObjectURL(combinedUrl);
          setCombinedFile(combined);
          setCombinedUrl(URL.createObjectURL(combined));
        }
      } finally {
        if (!cancelled) setCombiningCandidate(false);
      }
    }, 0);
    return () => { cancelled = true; clearTimeout(handle); };
  }, [segments]);

  const [combinedRefUrl, setCombinedRefUrl] = useState(null);
  const [combinedRefFile, setCombinedRefFile] = useState(null);
  const [combiningRef, setCombiningRef] = useState(false);
  useEffect(() => {
    let cancelled = false;
    if (!refSegments.length) { if (combinedRefUrl) URL.revokeObjectURL(combinedRefUrl); setCombinedRefUrl(null); setCombinedRefFile(null); return; }
    setCombiningRef(true);
    const handle = setTimeout(async () => {
      try {
        const combined = await combineSegmentsWav(refSegments);
        if (!cancelled && combined) {
          if (combinedRefUrl) URL.revokeObjectURL(combinedRefUrl);
          setCombinedRefFile(combined);
          setCombinedRefUrl(URL.createObjectURL(combined));
        }
      } finally {
        if (!cancelled) setCombiningRef(false);
      }
    }, 0);
    return () => { cancelled = true; clearTimeout(handle); };
  }, [refSegments]);

  const candidatePreviewUrl = combinedUrl;
  const referencePreviewUrl = combinedRefUrl;

  // Fetch model info (layer count) once; also warms up the model on the server
  useEffect(() => {
    fetch('/api/model_info')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && typeof d.num_layers === 'number') setNumLayers(d.num_layers);
        if (d && d.default_layer != null) setEmbLayer(String(d.default_layer));
      })
      .catch(() => {});
    fetch('/api/words')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d && Array.isArray(d.words)) setWords(d.words); })
      .catch(() => {});
  }, []);

  const selectedWordInfo = useMemo(
    () => words.find((w) => w.word === refWord.trim()) || null,
    [words, refWord]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Enter') { submit(); }
      if (e.key === 'Escape') { resetAll(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const playOneSecond = (url) => {
    if (!url) return;
    const audio = new Audio(url);
    audio.play();
    setTimeout(() => { audio.pause(); audio.currentTime = 0; }, 1000);
  };

  const submit = async () => {
    setError("");
    setScore(null);
    setExplain(null);
    const wordTyped = refWord.trim();
    if (wordTyped && !selectedWordInfo) {
      setError(`${t.wordUnknown}: ${wordTyped}`);
      push(t.wordUnknown, 'error');
      return;
    }
    if ((!refSegments.length && !selectedWordInfo) || !segments.length) {
      setError(t.pleaseBoth);
      push(t.pleaseBoth, 'error');
      return;
    }

    // Layer spec: "20" or "19,20,21"; empty = server default / final output
    const layerSpec = embLayer.trim();
    if (layerSpec && !/^\d+(\s*,\s*\d+)*$/.test(layerSpec)) {
      setError('Invalid layer spec — use e.g. 20 or 19,20,21');
      push('Invalid layer spec', 'error');
      return;
    }

    // Multi-template scoring: each segment is sent separately; the server
    // embeds each clip and averages them into a prototype per side. A library
    // word contributes all its recordings as additional reference templates.
    const form = new FormData();
    if (selectedWordInfo) form.append('reference_word', selectedWordInfo.word);
    refSegments.forEach((f) => form.append('reference', f));
    segments.forEach((f) => form.append('candidate', f));
    if (layerSpec) form.append('layer', layerSpec.replace(/\s+/g, ''));

    setLoading(true); setUploadProgress(5); push(t.uploading);
    try {
      const res = await fetch('/api/similarity', { method: 'POST', body: form });
      setUploadProgress(60);
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      setScore(data?.score ?? null); setExplain(data?.explain ?? null); setUploadProgress(100); push(t.computed, 'success');
    } catch (e) {
      setError(e.message || 'Failed to compute similarity.'); push(e.message || 'Failed to compute', 'error');
    } finally {
      setLoading(false); setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  // Theme toggle
  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    try { localStorage.setItem('theme', next ? 'dark' : 'light'); } catch {}
  };

  useEffect(() => {
    try {
      const saved = localStorage.getItem('theme');
      if (saved) {
        const isDark = saved === 'dark';
        setDark(isDark);
        document.documentElement.classList.toggle('dark', isDark);
      }
    } catch {}
  }, []);

  const pass = shownScore != null ? (shownScore >= threshold) : null;

  return (
    <main className="relative min-h-screen bg-gradient-to-b from-slate-50 via-slate-50 to-white dark:from-slate-950 dark:via-slate-950 dark:to-black font-sans bg-grain">
      {/* Glass header */}
      <div className="sticky top-0 z-40 backdrop-blur bg-white/70 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge className="border border-brand/50 bg-white text-brand dark:bg-transparent"><Waves className="w-3.5 h-3.5 mr-1"/> {t.uwBadge}</Badge>
            <Badge className="bg-slate-900 text-white dark:bg-slate-200 dark:text-slate-900">{t.modelBadge}</Badge>
          </div>
          <div className="flex items-center gap-2">
            <PageTabs page={page} onNavigate={onNavigate} />
            <Button variant="subtle" onClick={() => setLang((l) => l === 'EN' ? 'HMN' : 'EN')} className="px-3">{lang}</Button>
            <Button variant="subtle" onClick={toggleDark} className="px-3">{dark ? t.light : t.dark}</Button>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-10 sm:py-16">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start justify-between gap-6 mb-8">
          <div>
            <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white">{t.title}</h1>
            <p className="text-gray-600 dark:text-slate-300 mt-3 max-w-2xl leading-relaxed">{t.desc}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <Button variant="outline" onClick={resetAll}><Trash2 className="w-4 h-4"/>{t.reset}</Button>
            <Button onClick={submit} disabled={loading}><Headphones className="w-4 h-4"/>{loading ? t.scoring : t.compute}</Button>
          </div>
        </div>

        {/* Uploader Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Reference side */}
          <Card className="glow">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center"><Upload className="w-4 h-4 text-slate-600 dark:text-slate-300"/></div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{t.reference}</h2>
            </div>
            <div className="space-y-3">
              {/* Pick a word from the reference library */}
              {words.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      list="word-options"
                      value={refWord}
                      onChange={(e) => setRefWord(e.target.value)}
                      placeholder={t.wordPlaceholder}
                      className="flex-1 px-4 py-2 rounded-xl text-sm border border-gray-300 bg-white text-gray-700 dark:bg-slate-900 dark:border-slate-700 dark:text-slate-200 outline-none focus:ring-2 focus:ring-brand"
                    />
                    {refWord && (
                      <Button variant="ghost" size="icon" title={t.wordClear} onClick={() => setRefWord("")}><Trash2 className="w-4 h-4"/></Button>
                    )}
                  </div>
                  <datalist id="word-options">
                    {words.map((w) => (<option key={w.word} value={w.word} />))}
                  </datalist>
                  {selectedWordInfo && (
                    <div className="rounded-xl border dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900 glow space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium text-slate-700 dark:text-slate-200">“{selectedWordInfo.word}”</div>
                        <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">{selectedWordInfo.n_templates} {t.wordTemplates}</Badge>
                      </div>
                      <audio controls className="w-full" src={`/api/word_audio?word=${encodeURIComponent(selectedWordInfo.word)}`} />
                    </div>
                  )}
                  <p className="text-[11px] text-gray-500 dark:text-slate-500">{t.orUpload}</p>
                </div>
              )}

              {/* Add uploaded WAV as a new reference segment */}
              <FileInput id="ref-file" placeholder={t.chooseFile} onChange={(e) => { const f = e.target.files?.[0]; if (f) { setRefSegments((prev)=>[...prev, f]); push(t.refLoaded,'success'); } }} />

              {/* Reference segments list */}
              {refSegments.length > 0 && (
                <div className="space-y-2">
                  {refSegments.map((seg, i) => (
                    <div key={i} className="rounded-xl border dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900 glow">
                      <div className="flex items-center justify-between gap-3">
                        <EditableName name={seg.name} onCommit={(val) => commitRenameRefSegment(i, val)} className="flex-1" />
                        <div className="flex items-center gap-1">
                          <IconButtonLink href={refSegmentUrls[i]} filename={seg.name}>
                            <Download className="w-4 h-4" />
                          </IconButtonLink>
                          <Button variant="ghost" size="icon" onClick={() => setRefSegments((prev)=> prev.filter((_,idx)=> idx!==i))}><Trash2 className="w-4 h-4"/></Button>
                        </div>
                      </div>
                      <WaveCanvas file={seg} />
                      <audio controls className="w-full mt-2" src={refSegmentUrls[i]} />
                    </div>
                  ))}
                </div>
              )}

              {/* Combined reference preview */}
              {refSegments.length > 0 && (
                <div className="rounded-xl border dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900 glow">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-700 dark:text-slate-200 font-medium">Combined Reference</div>
                    {combiningRef && <div className="text-xs text-slate-500">Combining…</div>}
                    {combinedRefFile && !combiningRef && (
                      <IconButtonLink href={combinedRefUrl} filename={combinedRefFile.name}>
                        <Download className="w-4 h-4" />
                      </IconButtonLink>
                    )}
                  </div>
                  {!combiningRef && combinedRefUrl && <audio controls className="w-full mt-2" src={combinedRefUrl} />}
                </div>
              )}

              {refSegments.length === 0 && (
                <p className="text-xs text-gray-500 dark:text-slate-400 pt-1">{t.tip}</p>
              )}
            </div>
          </Card>

          {/* Candidate side (unchanged except adding WaveCanvas for each segment) */}
          <Card className="glow">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center"><Mic className="w-4 h-4 text-slate-600 dark:text-slate-300"/></div>
              <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{t.candidate}</h2>
            </div>
            <div className="space-y-3">
              <FileInput id="user-file" placeholder={t.chooseFile} onChange={(e) => { const f = e.target.files?.[0]; if (f) { setSegments((prev)=>[...prev, f]); push(t.candLoaded,'success'); } }} />

              <div className="flex items-center gap-3 pt-1">
                {recState === "idle" && (
                  <>
                    {!micReady && !micWarming && (
                      <Button variant="subtle" onClick={warmUpMic} title={labelPrepare} aria-label={labelPrepare}>{labelPrepare}</Button>
                    )}
                    {micWarming && (
                      <Button variant="subtle" disabled className="cursor-wait" title={labelPreparing} aria-label={labelPreparing}><Loader2 className="w-4 h-4 animate-spin"/>{labelPreparing}</Button>
                    )}
                    {micReady && !micWarming && (
                      <Button variant="subtle" disabled title={labelPrepared} aria-label={labelPrepared}>{labelPrepared}</Button>
                    )}
                    <Button variant="subtle" onClick={startRecording}><Mic className="w-4 h-4"/>{t.record}</Button>
                  </>
                )}
                {recState === "arming" && (
                  <Button variant="subtle" disabled className="cursor-wait"><Loader2 className="w-4 h-4 animate-spin"/>Starting{recCountdown > 0 ? ` in ${recCountdown}…` : '…'}</Button>
                )}
                {recState === "recording" && (
                  <Button variant="subtle" onClick={stopRecording} className="bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/40 dark:text-red-300"><Pause className="w-4 h-4"/>{t.stop} {recTime ? `${recTime}s` : ""}</Button>
                )}
                <div className="flex-1 h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  <div className="h-full bg-brand" style={{ width: `${Math.round(vu * 100)}%` }} />
                </div>
              </div>

              {segments.length > 0 && (
                <div className="space-y-2">
                  {segments.map((seg, i) => (
                    <div key={i} className="rounded-xl border dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900 glow">
                      <div className="flex items-center justify-between gap-3">
                        <EditableName name={seg.name} onCommit={(val) => commitRenameSegment(i, val)} className="flex-1" />
                        <div className="flex items-center gap-1">
                          <IconButtonLink href={segmentUrls[i]} filename={seg.name}>
                            <Download className="w-4 h-4" />
                          </IconButtonLink>
                          <Button variant="ghost" size="icon" onClick={() => setSegments((prev)=> prev.filter((_,idx)=> idx!==i))}><Trash2 className="w-4 h-4"/></Button>
                        </div>
                      </div>
                      <WaveCanvas file={seg} />
                      <audio controls className="w-full mt-2" src={segmentUrls[i]} />
                    </div>
                  ))}
                </div>
              )}

              {segments.length > 0 && (
                <div className="rounded-xl border dark:border-slate-800 p-3 bg-slate-50 dark:bg-slate-900 glow">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-700 dark:text-slate-200 font-medium">Combined Preview</div>
                    {combiningCandidate && <div className="text-xs text-slate-500">Combining…</div>}
                    {combinedFile && !combiningCandidate && (
                      <IconButtonLink href={combinedUrl} filename={combinedFile.name}>
                        <Download className="w-4 h-4" />
                      </IconButtonLink>
                    )}
                  </div>
                  {!combiningCandidate && combinedUrl && <audio controls className="w-full mt-2" src={combinedUrl} />}
                </div>
              )}

              <p className="text-[11px] text-gray-500 dark:text-slate-500">{t.recordNote}</p>
            </div>
          </Card>
        </div>

        {/* Threshold */}
        <div className="mt-8">
          <Card className="glow">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="text-base md:text-lg font-semibold text-slate-800 dark:text-slate-100">{t.decision}</div>
                <div className="text-xs text-gray-500 dark:text-slate-400">{t.decisionDesc}</div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-48">
                  <input type="range" min="0" max="1" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-brand" />
                </div>
                <div className="text-sm font-semibold tabular-nums text-slate-900 dark:text-white">{threshold.toFixed(2)}</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="text-base md:text-lg font-semibold text-slate-800 dark:text-slate-100">{t.layerTitle}</div>
                <div className="text-xs text-gray-500 dark:text-slate-400">{t.layerDesc}{numLayers ? ` (1–${numLayers})` : ''}</div>
              </div>
              <input
                type="text"
                inputMode="numeric"
                value={embLayer}
                onChange={(e) => setEmbLayer(e.target.value)}
                placeholder={t.layerFinal}
                className="w-32 px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white text-center tabular-nums outline-none focus:ring-2 focus:ring-brand"
              />
            </div>
          </Card>
        </div>

        {/* Result */}
        <div className="mt-8">
          <Card className="glow">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center"><Info className="w-4 h-4 text-slate-600 dark:text-slate-300"/></div>
              <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{t.similarity}</h3>
            </div>

            {loading && (
              <div className="space-y-3">
                <div className="h-9 w-40 bg-slate-200 dark:bg-slate-800 rounded-lg animate-pulse" />
                <div className="h-2 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />
                <div className="h-4 w-2/3 bg-slate-200 dark:bg-slate-800 rounded animate-pulse" />
              </div>
            )}

            {!loading && error && (
              <div className="text-sm text-red-600 dark:text-red-400">{error}</div>
            )}

            {!loading && shownScore != null && (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                <div className="flex items-baseline gap-3">
                    <div className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white tabular-nums">{fmtCos(shownScore)}</div>
                    <Badge className={cn("text-xs", shownScore >= 0.7 ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300" : shownScore >= 0.5 ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300")}>cosine</Badge>
                  </div>
                  <div className={cn("px-3 py-1 rounded-lg text-xs font-medium", pass ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300")}>{pass ? t.pass : t.below}</div>
                </div>
                <div>
                  <Progress value={100 * shownScore} />
                </div>
                {explain && (
                  <p className="text-sm text-gray-600 dark:text-slate-300 leading-relaxed">{explain}</p>
                )}

                {/* A/B quick listen */}
                <div className="flex items-center gap-2 pt-2">
                  <Button variant="outline" onClick={() => playOneSecond(referencePreviewUrl)}><Play className="w-4 h-4"/>{t.ref1s}</Button>
                  <Button variant="outline" onClick={() => playOneSecond(candidatePreviewUrl)}><Play className="w-4 h-4"/>{t.cand1s}</Button>
                </div>

                <p className="text-xs text-gray-500 dark:text-slate-400 pt-1">Score = cosine similarity ∈ [−1, 1].</p>
              </div>
            )}

            {!loading && shownScore == null && !error && (
              <p className="text-sm text-gray-600 dark:text-slate-300">{t.selectTwo}</p>
            )}
          </Card>
        </div>

        {/* Footer removed per request */}
      </div>

      <Toasts />
    </main>
  );
}
