// Minimal ZIP writer (STORE / no compression) — zero dependency.
// WAV PCM barely compresses, so "store" keeps it simple and fast.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function dosDateTime(d) {
  const time = ((d.getHours() & 0x1f) << 11) | ((d.getMinutes() & 0x3f) << 5) | ((d.getSeconds() >> 1) & 0x1f);
  const date = (((d.getFullYear() - 1980) & 0x7f) << 9) | (((d.getMonth() + 1) & 0xf) << 5) | (d.getDate() & 0x1f);
  return { time, date };
}

const enc = new TextEncoder();

/**
 * Build a ZIP Blob from a list of { name, data } entries.
 * data may be a string, ArrayBuffer, or Uint8Array.
 */
export function makeZip(files, when = new Date()) {
  const { time, date } = dosDateTime(when);
  const chunks = [];
  const central = [];
  let offset = 0;

  for (const f of files) {
    let data = f.data;
    if (typeof data === 'string') data = enc.encode(data);
    else if (data instanceof ArrayBuffer) data = new Uint8Array(data);
    const nameBytes = enc.encode(f.name);
    const crc = crc32(data);

    const lh = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(6, 0, true);
    lv.setUint16(8, 0, true); // method 0 = store
    lv.setUint16(10, time, true);
    lv.setUint16(12, date, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, data.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    lh.set(nameBytes, 30);
    chunks.push(lh, data);

    const ch = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, time, true);
    cv.setUint16(14, date, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    ch.set(nameBytes, 46);
    central.push(ch);

    offset += lh.length + data.length;
  }

  const centralStart = offset;
  let centralSize = 0;
  for (const c of central) { chunks.push(c); centralSize += c.length; }

  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, centralStart, true);
  chunks.push(eocd);

  return new Blob(chunks, { type: 'application/zip' });
}

// ---- Minimal ZIP reader (STORE + DEFLATE) ----
// Reads entries via the End-Of-Central-Directory + central directory records.
// STORE (method 0) is sliced directly; DEFLATE (method 8) is inflated with the
// browser's DecompressionStream. Returns [{ name, bytes: Uint8Array }].
export async function readZip(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  const dv = new DataView(arrayBuffer);
  // find EOCD (0x06054b50) scanning backwards
  let eocd = -1;
  for (let i = u8.length - 22; i >= 0; i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('Not a ZIP file (no end-of-central-directory)');
  const count = dv.getUint16(eocd + 10, true);
  let ptr = dv.getUint32(eocd + 16, true); // central dir offset
  const out = [];
  for (let n = 0; n < count; n++) {
    if (dv.getUint32(ptr, true) !== 0x02014b50) break;
    const method = dv.getUint16(ptr + 10, true);
    const compSize = dv.getUint32(ptr + 20, true);
    const nameLen = dv.getUint16(ptr + 28, true);
    const extraLen = dv.getUint16(ptr + 30, true);
    const commentLen = dv.getUint16(ptr + 32, true);
    const localOff = dv.getUint32(ptr + 42, true);
    const name = new TextDecoder().decode(u8.subarray(ptr + 46, ptr + 46 + nameLen));
    // local header: recompute data start (its name/extra lengths can differ)
    const lNameLen = dv.getUint16(localOff + 26, true);
    const lExtraLen = dv.getUint16(localOff + 28, true);
    const dataStart = localOff + 30 + lNameLen + lExtraLen;
    const raw = u8.subarray(dataStart, dataStart + compSize);
    let bytes;
    if (method === 0) {
      bytes = raw.slice();
    } else if (method === 8) {
      const ds = new DecompressionStream('deflate-raw');
      const buf = await new Response(new Blob([raw]).stream().pipeThrough(ds)).arrayBuffer();
      bytes = new Uint8Array(buf);
    } else {
      throw new Error(`Unsupported ZIP compression method ${method} for ${name}`);
    }
    if (!name.endsWith('/')) out.push({ name, bytes });
    ptr += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}
