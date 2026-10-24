import { useCallback, useRef, useState } from 'react';

// Encode mono Float32 PCM samples to a 16-bit WAV byte array.
export function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);   // PCM
  view.setUint16(22, 1, true);   // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let idx = 44;
  for (let i = 0; i < samples.length; i++, idx += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

export function wavBlob(samples, sampleRate) {
  return new Blob([encodeWAV(samples, sampleRate)], { type: 'audio/wav' });
}

// Peak amplitude, used to flag near-silent (clipped-window) recordings.
export function peakLevel(samples) {
  let p = 0;
  for (let i = 0; i < samples.length; i++) { const a = Math.abs(samples[i]); if (a > p) p = a; }
  return p;
}

/**
 * Microphone recorder hook with imperative control, suitable for both
 * click-to-record and timed (guided) capture.
 *   armMic()      – request permission + spin up the audio graph (once/session)
 *   startCapture()– begin buffering samples
 *   stopCapture() – stop buffering, returns { samples, sampleRate }
 *   releaseMic()  – free the device
 * Exposes { micReady, vu } for UI.
 */
export function useRecorder() {
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const procRef = useRef(null);
  const gainRef = useRef(null);
  const inputGainRef = useRef(null);
  const vuRafRef = useRef(null);
  const buffersRef = useRef([]);
  const capturingRef = useRef(false);
  const clipRef = useRef(false);
  const [micReady, setMicReady] = useState(false);
  const [vu, setVu] = useState(0);
  const [inputGain, setInputGainState] = useState(1);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');

  // Input gain is applied in the audio graph, so it affects what gets recorded
  // (browsers give no control over the OS input level). Kept out of React
  // state on the audio side so changing it never rebuilds the graph.
  const setInputGain = useCallback((v) => {
    const g = Math.max(0.1, Math.min(10, Number(v) || 1));
    setInputGainState(g);
    if (inputGainRef.current) inputGainRef.current.gain.value = g;
  }, []);

  const armMic = useCallback(async (requestedDeviceId) => {
    if (ctxRef.current) return;
    const id = requestedDeviceId || deviceId;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
        ...(id ? { deviceId: { exact: id } } : {}),
      },
    });
    streamRef.current = stream;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const source = ctx.createMediaStreamSource(stream);

    // source -> inputGain -> processor (what we record) and -> analyser (what
    // the level meter shows), so the meter reflects the gain being applied.
    const inputGainNode = ctx.createGain();
    inputGainNode.gain.value = inputGain;
    source.connect(inputGainNode);

    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!capturingRef.current) return;
      const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
      for (let i = 0; i < chunk.length; i++) {
        if (chunk[i] > 0.99 || chunk[i] < -0.99) { clipRef.current = true; break; }
      }
      buffersRef.current.push(chunk);
    };
    const gain = ctx.createGain();
    gain.gain.value = 0;  // silent monitor: keeps the processor pulling audio
    inputGainNode.connect(processor);
    processor.connect(gain);
    gain.connect(ctx.destination);

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    inputGainNode.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const loop = () => {
      analyser.getByteFrequencyData(data);
      let a = 0;
      for (let i = 0; i < data.length; i++) a += data[i];
      setVu(a / data.length / 255);
      vuRafRef.current = requestAnimationFrame(loop);
    };
    loop();

    ctxRef.current = ctx;
    procRef.current = processor;
    gainRef.current = gain;
    inputGainRef.current = inputGainNode;
    setMicReady(true);

    // Device labels are only populated once permission has been granted.
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      setDevices(list.filter((d) => d.kind === 'audioinput'));
      const track = stream.getAudioTracks()[0];
      const active = track && track.getSettings ? track.getSettings().deviceId : '';
      if (active) setDeviceId(active);
    } catch {}
  }, [deviceId, inputGain]);

  const startCapture = useCallback(() => {
    buffersRef.current = [];
    clipRef.current = false;
    capturingRef.current = true;
  }, []);

  const stopCapture = useCallback(() => {
    capturingRef.current = false;
    const buffers = buffersRef.current;
    buffersRef.current = [];
    let total = 0;
    for (const b of buffers) total += b.length;
    const samples = new Float32Array(total);
    let off = 0;
    for (const b of buffers) { samples.set(b, off); off += b.length; }
    const sr = ctxRef.current ? ctxRef.current.sampleRate : 44100;
    return { samples, sampleRate: sr, clipped: clipRef.current };
  }, []);

  const releaseMic = useCallback(() => {
    if (vuRafRef.current) cancelAnimationFrame(vuRafRef.current);
    try { if (procRef.current) { procRef.current.onaudioprocess = null; procRef.current.disconnect(); } } catch {}
    try { if (gainRef.current) gainRef.current.disconnect(); } catch {}
    try { if (inputGainRef.current) inputGainRef.current.disconnect(); } catch {}
    try { if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop()); } catch {}
    try { if (ctxRef.current) ctxRef.current.close(); } catch {}
    ctxRef.current = null; procRef.current = null; gainRef.current = null;
    inputGainRef.current = null; streamRef.current = null;
    capturingRef.current = false;
    setMicReady(false);
    setVu(0);
  }, []);

  // Switching device needs a fresh graph: tear down, then arm the new one.
  const selectDevice = useCallback(async (id) => {
    setDeviceId(id);
    const wasReady = !!ctxRef.current;
    releaseMic();
    if (wasReady) await armMic(id);
  }, [armMic, releaseMic]);

  return {
    armMic, startCapture, stopCapture, releaseMic, micReady, vu,
    inputGain, setInputGain, devices, deviceId, selectDevice,
  };
}
