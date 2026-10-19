export const COLOR = { fg: '#ece9e2', muted: '#868d97', blue: '#58c4dd', yellow: '#f2cf4a', red: '#fc6255', green: '#83c167' };
const RGB = { blue: [88, 196, 221], yellow: [242, 207, 74], red: [252, 98, 85], green: [131, 193, 103] };
const MASKED = [38, 41, 47];

export const rgba = (c, a) => `rgba(${RGB[c]},${a})`;

// mode: 'v' values (blue/red, or blue alone if all >= 0), 'g' gradients, 'u' updates.
export function paint(cv, data, rows, cols, mode = 'v') {
  cv.width = cols;
  cv.height = rows;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(cols, rows);
  let max = 1e-12, signed = false;
  for (const v of data) if (Number.isFinite(v)) { max = Math.max(max, Math.abs(v)); signed ||= v < 0; }
  data.forEach((v, k) => {
    const a = Math.abs(v) / max;
    const c = !Number.isFinite(v) ? MASKED : mode === 'g' ? RGB.yellow : mode === 'u' ? RGB.green : signed && v < 0 ? RGB.red : RGB.blue;
    for (let q = 0; q < 3; q++) img.data[k * 4 + q] = Number.isFinite(v) ? 4 + c[q] * a : c[q];
    img.data[k * 4 + 3] = 255;
  });
  ctx.putImageData(img, 0, 0);
  return cv;
}

export function heatCanvas(data, rows, cols, mode, maxW, maxH) {
  const cv = paint(document.createElement('canvas'), data, rows, cols, mode);
  const cell = Math.max(1, Math.min(10, Math.floor(maxW / cols), Math.floor(maxH / rows)));
  cv.style.width = `${cols * cell}px`;
  cv.style.height = `${rows * cell}px`;
  return cv;
}

export function fitCanvas(cv) {
  const dpr = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

// A trace's flat buffer, split into per-tensor value / grad / post-update slices.
export function decode(trace) {
  let off = 0;
  const take = n => trace.buf.subarray(off, off += n);
  const tensors = trace.tensors.map(t => {
    const [rows, cols] = t.view;
    return { ...t, rows, cols, value: take(rows * cols), grad: take(rows * cols) };
  });
  if (trace.after) for (const t of tensors) if (t.param) t.after = take(t.rows * t.cols);
  return { ...trace, tensors };
}
