// The Architecture view: the whole GPT, left to right, with forward and backward pulses each step.
import { COLOR, fitCanvas, rgba } from './heat.js';

const W = 1310, H = 215, STREAM = 160, BRANCH = 62, SPAN = 185, CYCLE_MS = 1100;
const SUBLAYERS = [['b0.attn', 'b0.ln1', 140], ['b0.mlp', 'b0.ln2', 345], ['b1.attn', 'b1.ln1', 575], ['b1.mlp', 'b1.ln2', 780]];
const BOXES = [
  { id: 'emb', x: 82, y: STREAM, w: 78, h: 30, label: 'embed' },
  ...SUBLAYERS.flatMap(([main, ln, x0]) => [
    { id: ln, x: x0 + 38, y: BRANCH, w: 54, h: 24, label: ln.endsWith('1') ? 'ln_1' : 'ln_2' },
    { id: main, x: x0 + 122, y: BRANCH, w: 100, h: 50, label: main.endsWith('attn') ? 'attn' : 'mlp', big: true },
  ]),
  { id: 'lnf', x: 1045, y: STREAM, w: 64, h: 26, label: 'ln_f' },
  { id: 'head', x: 1140, y: STREAM, w: 100, h: 30, label: 'head · wteᵀ' },
  { id: 'loss', x: 1250, y: STREAM, w: 94, h: 30, label: 'loss' },
];
const branch = x0 => [[x0, STREAM], [x0, BRANCH], [x0 + SPAN, BRANCH], [x0 + SPAN, STREAM]];

function along(pts, u) {
  const lens = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
  let d = u * lens.reduce((a, b) => a + b);
  for (let i = 0; i < lens.length; i++) {
    if (d <= lens[i]) return pts[i].map((v, k) => v + (pts[i + 1][k] - v) * d / lens[i]);
    d -= lens[i];
  }
  return pts.at(-1);
}

export function createArch(cv, onSelect) {
  let phase = 0;

  const hit = e => {
    const r = cv.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
    return BOXES.find(b => Math.abs(x - b.x) <= b.w / 2 + 4 && Math.abs(y - b.y) <= b.h / 2 + 4);
  };
  cv.addEventListener('click', e => hit(e) && onSelect(hit(e).id));
  cv.addEventListener('mousemove', e => { cv.style.cursor = hit(e) ? 'pointer' : 'default'; });

  return function draw(s, dt) {
    if (s.running) phase = (phase + dt / CYCLE_MS) % 1;
    const [ctx, w, h] = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.scale(w / W, h / H);
    ctx.textBaseline = 'middle';
    ctx.font = '11px "IBM Plex Mono", monospace';

    [[0, 122, 545], [1, 557, 980]].forEach(([b, x0, x1]) => {
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = '#2c3037';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, 22, x1 - x0, 178);
      ctx.setLineDash([]);
      ctx.fillStyle = COLOR.muted;
      ctx.textAlign = 'left';
      ctx.fillText(`block ${b}`, x0 + 8, 188);
    });
    ctx.strokeStyle = '#3d434c';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(14, STREAM);
    ctx.lineTo(1300, STREAM);
    ctx.stroke();
    ctx.fillStyle = COLOR.muted;
    ctx.fillText('idx', 4, STREAM - 16);
    ctx.textAlign = 'center';
    ctx.fillText(`residual stream · ${s.info.B} × ${s.info.T} × 64`, 450, STREAM + 26);
    for (const [, , x0] of SUBLAYERS) {
      ctx.strokeStyle = '#3d434c';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      branch(x0).forEach(p => ctx.lineTo(...p));
      ctx.stroke();
      const ax = x0 + SPAN;
      ctx.beginPath();
      ctx.arc(ax, STREAM, 8, 0, 7);
      ctx.fillStyle = '#000';
      ctx.fill();
      ctx.strokeStyle = COLOR.muted;
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ax - 5, STREAM);
      ctx.lineTo(ax + 5, STREAM);
      ctx.moveTo(ax, STREAM - 5);
      ctx.lineTo(ax, STREAM + 5);
      ctx.stroke();
    }

    const fx = Math.min(1, phase / 0.45) * W, bx = W - Math.max(0, Math.min(1, (phase - 0.5) / 0.45)) * W;
    const norms = s.norms.at(-1) ?? {}, max = Math.max(1e-9, ...Object.values(norms));
    for (const b of BOXES) {
      const g = b.id === 'loss' ? 1 : Math.sqrt((norms[b.id] ?? 0) / max), sel = b.id === s.sel.component;
      const near = s.running && phase > 0.5 ? Math.max(0, 1 - Math.abs(bx - b.x) / 40) : 0;
      const x0 = b.x - b.w / 2, y0 = b.y - b.h / 2;
      if (sel) {
        ctx.strokeStyle = COLOR.fg;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(b.x, y0 + b.h);
        ctx.lineTo(b.x, H);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.shadowColor = COLOR.yellow;
      ctx.shadowBlur = 20 * near * g;
      ctx.fillStyle = '#07080a';
      ctx.beginPath();
      ctx.roundRect(x0, y0, b.w, b.h, 6);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = sel ? COLOR.fg : rgba('yellow', 0.18 + 0.7 * Math.max(g * 0.75, near));
      ctx.lineWidth = sel ? 2 : 1.3;
      ctx.stroke();
      ctx.fillStyle = COLOR.fg;
      ctx.font = `500 ${b.big ? 13 : 11}px "IBM Plex Mono", monospace`;
      if (!b.big) {
        ctx.textAlign = 'center';
        ctx.fillText(b.id === 'loss' && s.losses.length ? `loss ${s.losses.at(-1).toFixed(2)}` : b.label, b.x, b.y + 1);
        continue;
      }
      ctx.textAlign = 'left';
      ctx.fillText(b.label, x0 + 9, b.y - 9);
      ctx.fillStyle = COLOR.muted;
      ctx.font = '10px "IBM Plex Mono", monospace';
      if (norms[b.id] !== undefined) ctx.fillText(`‖g‖ ${norms[b.id].toFixed(2)}`, x0 + 9, b.y + 11);
      const thumb = s.attention[b.id];
      if (thumb) {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(thumb, x0 + b.w - 42, y0 + 7, 36, 36);
      } else if (!b.id.endsWith('attn')) {
        const hist = s.norms.slice(-60).map(n => n[b.id]), top = Math.max(1e-9, ...hist);
        ctx.beginPath();
        hist.forEach((v, i) => ctx.lineTo(x0 + b.w - 46 + i / 59 * 40, y0 + 40 - v / top * 28));
        ctx.strokeStyle = rgba('yellow', 0.85);
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    if (s.running) {
      const dot = (x, y, c) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 7);
        ctx.fillStyle = ctx.shadowColor = c;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      };
      const pulse = (x, c) => {
        dot(x, STREAM, c);
        for (const [, , x0] of SUBLAYERS) if (x >= x0 && x <= x0 + SPAN) dot(...along(branch(x0), (x - x0) / SPAN), c);
      };
      if (phase < 0.47) pulse(fx, COLOR.blue);
      if (phase > 0.5 && phase < 0.97) pulse(bx, COLOR.yellow);
    }
    ctx.restore();
  };
}
