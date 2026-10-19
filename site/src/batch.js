// The Batch as text, and the (B, T, C) tensor it becomes as a 3D block of slices.
import { COLOR, fitCanvas, paint } from './heat.js';

const VIEW_TOKENS = 16;
const show = ch => ch === '\n' ? '↵' : ch;

export function createBatch(el, hoverEl, onRow) {
  let batch = null;

  el.addEventListener('click', e => {
    const row = e.target.closest('.r');
    if (row) onRow(+row.dataset.r);
  });
  el.addEventListener('mouseover', e => {
    el.querySelectorAll('.hov, .tgt').forEach(c => c.classList.remove('hov', 'tgt'));
    const c = e.target.closest('.c');
    if (!c) return;
    const r = +c.parentNode.dataset.r, j = +c.dataset.j, ch = batch[r][j], next = batch[r][j + 1];
    c.classList.add('hov');
    c.nextSibling?.classList.add('tgt');
    hoverEl.textContent = `idx[${r}, ${j}] = '${show(ch)}'  ·  target = '${show(next)}'`;
  });
  el.addEventListener('mouseleave', () => {
    el.querySelectorAll('.hov, .tgt').forEach(c => c.classList.remove('hov', 'tgt'));
    hoverEl.innerHTML = '&nbsp;';
  });

  return function render(rows, followed) {
    batch = rows;
    el.replaceChildren(...rows.map((win, r) => {
      const row = document.createElement('div');
      row.className = r === followed ? 'r on' : 'r';
      row.dataset.r = r;
      row.innerHTML = `<span class="n">${r}</span>`;
      [...win.slice(0, -1)].forEach((ch, j) => {
        const c = document.createElement('span');
        c.className = `c${j < VIEW_TOKENS ? ' v' : ''}${ch === '\n' ? ' nl' : ''}`;
        c.dataset.j = j;
        c.textContent = show(ch);
        row.append(c);
      });
      row.insertAdjacentHTML('beforeend', '<span class="tag">← shown below</span>');
      return row;
    }));
  };
}

export function createBlock(cv) {
  const face = document.createElement('canvas');
  let at = 0;

  // `slice` is the followed row's first VIEW_TOKENS tokens x channels, drawn tokens-across.
  function setSlice(t) {
    const out = new Float32Array(t.rows * t.cols);
    for (let i = 0; i < t.rows; i++) for (let j = 0; j < t.cols; j++) out[j * t.rows + i] = t.value[i * t.cols + j];
    paint(face, out, t.cols, t.rows);
  }

  function draw(B, T, row) {
    at += (row - at) * 0.18;
    const [ctx, w, h] = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const F = w * 0.5, dx = w * 0.3 / B, dy = -w * 0.2 / B, ox = w * 0.14, oy = h * 0.86;
    const pos = b => [ox + b * dx, oy - F + b * dy];
    const [fx, fy] = pos(0), [bx, by] = pos(B - 1);

    ctx.strokeStyle = 'rgba(134,141,151,.5)';
    ctx.lineWidth = 1;
    for (const [u, v] of [[0, 0], [F, 0], [F, F], [0, F]]) {
      ctx.beginPath();
      ctx.moveTo(fx + u, fy + v);
      ctx.lineTo(bx + u, by + v);
      ctx.stroke();
    }
    for (let b = B - 1; b >= 0; b--) {
      const [x, y] = pos(b);
      ctx.fillStyle = 'rgba(88,196,221,0.035)';
      ctx.fillRect(x, y, F, F);
      ctx.strokeStyle = 'rgba(134,141,151,0.28)';
      ctx.strokeRect(x, y, F, F);
      if (b !== Math.round(at)) continue;
      const [sx, sy] = pos(at), vw = F * VIEW_TOKENS / T;
      ctx.shadowColor = COLOR.blue;
      ctx.shadowBlur = 16;
      ctx.fillStyle = '#000';
      ctx.fillRect(sx, sy, F, F);
      ctx.shadowBlur = 0;
      ctx.imageSmoothingEnabled = false;
      if (face.width) ctx.drawImage(face, sx, sy, vw, F);
      ctx.strokeStyle = COLOR.blue;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx, sy, F, F);
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(sx, sy, vw, F);
      ctx.setLineDash([]);
      ctx.lineWidth = 1;
    }
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`tokens · ${T}`, fx + F / 2, fy + F + 6);
    ctx.fillStyle = COLOR.blue;
    ctx.textAlign = 'left';
    ctx.fillText('shown below', fx, fy + F + 22);
    ctx.save();
    ctx.translate(fx - 10, fy + F / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('channels · 64', 0, 0);
    ctx.restore();
    ctx.fillStyle = COLOR.muted;
    ctx.textBaseline = 'bottom';
    ctx.fillText(`batch · ${B}`, bx + F - 30, by - 4);
  }

  return { setSlice, draw };
}
