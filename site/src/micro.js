// The Micro view: one component's ops, played forward, backward and through the AdamW update.
import { heatCanvas, paint } from './heat.js';
import { ADAMW, BACKWARD, COMPONENTS, FORWARD, backwardKey, typeOf } from './source.js';

const ATTN_OP = 16;  // e / Σe: the attention pattern
const PHASE = { F: 'forward', B: 'backward', U: 'update' };
const OP_MS = 420, UPDATE_MS = 1400;

export const sameView = (a, b) => a.component === b.component && a.row === b.row && a.head === b.head;

export function attentionThumb(trace) {
  if (typeOf(trace.component) !== 'attn') return null;
  const t = trace.tensors[trace.ops[ATTN_OP].out], n = t.rows, out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) out.set(t.value.subarray(i * t.cols, i * t.cols + n), i * n);
  return paint(document.createElement('canvas'), out, n, n);
}

function sequence(trace) {
  const fwd = trace.ops.map((op, i) => ({ ph: 'F', i, op }));
  return [...fwd, ...fwd.toReversed().map(s => ({ ...s, ph: 'B' })), ...(trace.after ? [{ ph: 'U' }] : [])];
}

function labels(trace) {
  const names = trace.tensors.map(t => t.param?.split('.').slice(-2).join('.'));
  trace.ops.forEach((op, i) => { names[op.out] = FORWARD[typeOf(trace.component)].ops[i][1]; });
  return names.map((n, i) => n ?? (i === trace.ops[0].in[0] ? 'x' : trace.tensors[i].shape.length === 4 ? 'mask' : 'const'));
}

export function createMicro(el, sel, { onInteract, onHead, onPassStart }) {
  const heads = {};
  let trace = null, seq = [], names = [], pending = null, k = 0, p = 0, shown = '';

  function fig(data, t, mode, name, caption) {
    const d = document.createElement('div');
    d.className = 'fig';
    const cv = heatCanvas(data, t.rows, t.cols, mode, 260, 180);
    const cap = document.createElement('span');
    cap.className = 'cap';
    cap.textContent = caption ?? name;
    d.append(cv, cap);
    cv.onmousemove = e => {
      const r = cv.getBoundingClientRect();
      const j = Math.min(t.cols - 1, Math.floor((e.clientX - r.left) / r.width * t.cols));
      const i = Math.min(t.rows - 1, Math.floor((e.clientY - r.top) / r.height * t.rows));
      const v = data[i * t.cols + j], tok = t.shape.length >= 3 ? `  ·  token '${trace.batch[trace.row][i]}'` : '';
      el.hover.textContent = `${name}[${i}, ${j}] = ${Number.isFinite(v) ? v.toFixed(4) : '−inf'}${tok}`;
    };
    cv.onmouseleave = () => { el.hover.innerHTML = '&nbsp;'; };
    return d;
  }

  function sym(text) {
    const s = document.createElement('span');
    s.className = 'sym';
    s.textContent = text;
    return s;
  }

  function renderBig(s) {
    const T = i => trace.tensors[i];
    if (s.ph === 'U') {
      const params = trace.tensors.flatMap((t, i) => (t.after ? [i] : []));
      const i = params.find(i => T(i).rows > 1) ?? params[0], t = T(i);
      const delta = t.value.map((v, j) => v - t.after[j]);
      return [fig(t.value, t, 'v', names[i], `${names[i]} · before`), sym('−'),
        fig(delta, t, 'u', 'Δ', 'lr · m̂ / (√v̂ + ε) + decay'), sym('='), fig(t.after, t, 'v', names[i], 'after')];
    }
    const { op } = s, out = T(op.out);
    if (s.ph === 'B') {
      return [fig(out.grad, out, 'g', `∂${names[op.out]}`), sym('→'),
        ...op.in.flatMap((i, n) => [...(n ? [sym('+')] : []), fig(T(i).grad, T(i), 'g', `∂${names[i]}`)])];
    }
    const ins = op.in.map(i => fig(T(i).value, T(i), 'v', names[i]));
    const middle = ins.length === 2 ? [ins[0], sym(op.op), ins[1]] : [ins[0], sym(op.op === '*' ? '× c' : op.op)];
    return [...middle, sym('→'), fig(out.value, out, 'v', names[op.out], `${names[op.out]} · ${out.shape.join(' × ')}`)];
  }

  function code(s) {
    if (s.ph === 'U') return [ADAMW.fn, ADAMW.lines, ADAMW.hl];
    if (s.ph === 'F') {
      const src = FORWARD[typeOf(trace.component)];
      return [src.fn, src.lines, [src.ops[s.i][0]]];
    }
    const [name, lines] = BACKWARD[backwardKey(s.op)];
    return [`matrixgrad/engine.py · ${name} · _backward`, ['def _backward():', ...lines.map(l => `    ${l}`)], lines.map((_, n) => n + 1)];
  }

  function renderCode(s) {
    const [fn, lines, hl] = code(s);
    el.codeFn.textContent = fn;
    el.code.replaceChildren(...lines.map((line, n) => {
      const d = document.createElement('div');
      d.textContent = line || ' ';
      if (hl.includes(n)) d.className = s.ph;
      return d;
    }));
  }

  function renderHeads() {
    el.heads.hidden = typeOf(trace.component) !== 'attn';
    if (el.heads.hidden) return;
    el.heads.replaceChildren(Object.assign(document.createElement('span'), { className: 'cap', textContent: 'heads' }), ...[0, 1, 2, 3].map(h => {
      const b = document.createElement('button');
      b.className = h === trace.head ? 'hd on' : 'hd';
      const thumb = heads[`${trace.component}:${trace.row}:${h}`], cv = document.createElement('canvas');
      if (thumb) {
        [cv.width, cv.height] = [thumb.width, thumb.height];
        cv.getContext('2d').drawImage(thumb, 0, 0);
      }
      b.append(cv, `head ${h}`);
      b.onclick = () => onHead(h);
      return b;
    }));
  }

  function apply(t) {
    trace = t;
    seq = sequence(t);
    names = labels(t);
    pending = null;
    k = p = 0;
    shown = '';
    const thumb = attentionThumb(t);
    if (thumb) heads[`${t.component}:${t.row}:${t.head}`] = thumb;
    const [title, sub] = COMPONENTS[t.component], nF = t.ops.length;
    el.title.textContent = title;
    el.sub.textContent = `${sub} · step ${t.step} · row ${t.row}, first 16 tokens${thumb ? `, head ${t.head}` : ''}`;
    el.segs.replaceChildren(...seq.map(s => Object.assign(document.createElement('div'), { className: `seg ${s.ph}` })));
    el.names.innerHTML = `<span style="flex:${nF}">forward</span><span style="flex:${nF}">backward</span>${t.after ? '<span style="flex:3;text-align:right">update</span>' : ''}`;
    el.strip.replaceChildren(...t.ops.map((op, i) => {
      const b = document.createElement('button');
      b.className = 'chip';
      const out = t.tensors[op.out];
      b.append(heatCanvas(out.value, out.rows, out.cols, 'v', 48, 32), op.op);
      b.onclick = () => { onInteract(); k = i; p = 1; };
      return b;
    }));
    renderHeads();
    onPassStart();
  }

  function seek(clientX) {
    const r = el.tl.getBoundingClientRect(), u = Math.max(0, Math.min(0.9999, (clientX - r.left) / r.width));
    const w = seq.map(s => (s.ph === 'U' ? 3 : 1)), total = w.reduce((a, b) => a + b);
    let acc = 0;
    k = w.findIndex(x => (acc += x / total) > u);
    p = 1 - (acc - u) * total / w[k];
  }
  el.tl.onpointerdown = e => { el.tl.setPointerCapture(e.pointerId); onInteract(); seek(e.clientX); };
  el.tl.onpointermove = e => { if (el.tl.hasPointerCapture(e.pointerId)) seek(e.clientX); };
  el.tl.onkeydown = e => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!d || !trace) return;
    onInteract();
    k = Math.max(0, Math.min(seq.length - 1, k + d));
    p = 1;
    e.preventDefault();
  };

  return {
    receive(t) {
      if (!sameView(t, sel)) return;
      if (trace && sameView(trace, sel)) pending = t;
      else apply(t);
    },
    tick(dt, running) {
      if (!trace || !running) return;
      p += dt / (seq[k].ph === 'U' ? UPDATE_MS : OP_MS);
      while (p >= 1) {
        p -= 1;
        if (++k < seq.length) continue;
        if (pending) return apply(pending);
        k = 0;
        onPassStart();
      }
    },
    draw() {
      if (!trace) return;
      const s = seq[k], key = `${trace.step}:${trace.component}:${trace.row}:${trace.head}:${k}`;
      if (key !== shown) {
        shown = key;
        el.big.replaceChildren(...renderBig(s));
        renderCode(s);
        [...el.strip.children].forEach((c, i) => {
          c.className = `chip${s.ph !== 'F' || i < s.i ? ' done' : ''}${s.i === i && s.ph !== 'U' ? ` ${s.ph}` : ''}`;
          if (s.i === i && (c.offsetLeft < el.strip.scrollLeft || c.offsetLeft + c.offsetWidth > el.strip.scrollLeft + el.strip.clientWidth)) {
            el.strip.scrollLeft = c.offsetLeft - (el.strip.clientWidth - c.offsetWidth) / 2;
          }
        });
        const what = s.ph === 'U' ? 'AdamW step on this component' : `${s.ph === 'B' ? '∂ through ' : ''}${names[s.op.out]}`;
        el.status.innerHTML = `<b>${PHASE[s.ph]}</b> · ${what}`;
      }
      const w = seq.map(x => (x.ph === 'U' ? 3 : 1)), total = w.reduce((a, b) => a + b);
      el.mark.style.left = `${(w.slice(0, k).reduce((a, b) => a + b, 0) + w[k] * Math.min(1, p)) / total * 100}%`;
      [...el.segs.children].forEach((d, i) => d.classList.toggle('done', i < k || (i === k && p > 0)));
    },
  };
}
