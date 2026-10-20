import { createArch } from './arch.js';
import { createBatch, createBlock } from './batch.js';
import { createGenerate } from './generate.js';
import { COLOR, decode, fitCanvas } from './heat.js';
import { attentionThumb, createMicro, sameView } from './micro.js';

const $ = id => document.getElementById(id);
const AUTO_PAUSE_TOKENS = 500_000;
const HISTORY = 80;

const worker = new Worker('worker.js');
const recorded = fetch('recorded/run.json').then(r => r.json());
let redraw = 0, generateVisible = false;

const state = {
  info: null, ready: false, running: false, generating: false, resumeAfterGenerate: false, step: 0,
  losses: [], norms: [], attention: {}, batch: null,
  sel: { component: 'b0.attn', row: 0, head: 0 },
};

const arch = createArch($('arch'), component => select({ component }));
const renderBatch = createBatch($('batch'), $('batch-hover'), row => select({ row }));
const block = createBlock($('block'));
const micro = createMicro({
  title: $('m-title'), sub: $('m-sub'), heads: $('heads'), strip: $('strip'), big: $('big'), hover: $('hover'),
  tl: $('tl'), names: $('tl-names'), segs: $('segs'), mark: $('mark'), status: $('m-status'), codeFn: $('code-fn'), code: $('code'),
}, state.sel, {
  onInteract: pause,
  onHead: head => select({ head }),
  onPassStart: () => state.running && requestTrace(),
});

const generate = createGenerate({
  form: $('prompt'), input: $('prompt-text'), button: $('gen-btn'), out: $('gen-out'), bars: $('bars'), stat: $('gen-stat'),
}, worker, {
  onStart() {
    state.resumeAfterGenerate = state.running;
    state.generating = true;
    pause();
  },
  onEnd() {
    state.generating = false;
    if (state.resumeAfterGenerate) start();
    render();
  },
});

new IntersectionObserver(([e]) => {
  generateVisible = e.isIntersecting;
  if (generateVisible && state.ready) generate.load();
}, { rootMargin: '600px' }).observe($('generate'));

function render() {
  $('train-btn').disabled = !state.ready || state.generating;
  $('train-btn').textContent = !state.ready ? 'Starting engine…'
    : state.generating ? 'Paused for Generate' : state.running ? 'Pause' : state.step ? 'Keep training' : 'Train';
  redraw = 30;
}

function start() {
  state.running = true;
  worker.postMessage({ type: 'run' });
  render();
}

function pause() {
  state.running = false;
  worker.postMessage({ type: 'pause' });
  render();
}

function requestTrace() {
  if (state.ready) worker.postMessage({ type: state.running ? 'trace' : 'traceNow', ...state.sel });
}

function select(change) {
  Object.assign(state.sel, change);
  if (state.batch) renderBatch(state.batch, state.sel.row);
  if (!state.ready) $('m-status').textContent = 'starting the engine…';
  requestTrace();
  render();
}

function onStep(m) {
  const { B, T } = state.info, tokens = m.step * B * T;
  state.step = m.step;
  state.losses.push(m.loss);
  state.norms = [...state.norms.slice(1 - HISTORY), m.norms];
  $('status').textContent = `step ${m.step} · loss ${m.loss.toFixed(3)} · ${(tokens / 1e6).toFixed(2)}M tokens · ${Math.round(B * T / m.ms * 1000).toLocaleString()} tokens/s`;
  if (tokens >= AUTO_PAUSE_TOKENS && tokens - B * T < AUTO_PAUSE_TOKENS) {
    pause();
    recorded.then(rec => showLongRun(rec.longRun));
  }
  redraw = 30;
}

function onTrace(raw) {
  const t = decode(raw);
  const thumb = attentionThumb(t);
  if (thumb && t.head === 0) state.attention[t.component] = thumb;
  if (t.row === state.sel.row) {
    state.batch = t.batch;
    renderBatch(t.batch, t.row);
    block.setSlice(t.tensors.find(x => x.shape.length === 3 && x.shape[0] === state.info.B));
  }
  $('batch-cap').textContent = `idx · ${state.info.B} × ${state.info.T} · the batch at step ${t.step} · click a row to follow it`;
  $('block-cap').textContent = `x · ${state.info.B} × ${state.info.T} × 64`;
  micro.receive(t);
  redraw = 30;
}

function onSample(m) {
  $('sample-cap').textContent = `sample · step ${m.step}`;
  $('sample').textContent = m.text;
}

function showLongRun(text) {
  $('sample').textContent += `\n\n— after 2M tokens, from a longer run —\n${text}`;
}

function drawLoss() {
  const [ctx, w, h] = fitCanvas($('loss')), steps = Math.max(1000, state.step);
  const L = 30, R = 8, top = 8, bottom = 22, x = s => L + s / steps * (w - L - R), y = v => top + (4.5 - v) / 3.5 * (h - top - bottom);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.fillStyle = COLOR.muted;
  ctx.strokeStyle = '#1b1e23';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of [1, 2, 3, 4]) {
    ctx.beginPath();
    ctx.moveTo(L, y(v));
    ctx.lineTo(w - R, y(v));
    ctx.stroke();
    ctx.fillText(v, L - 6, y(v));
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= steps; s += steps / 4) ctx.fillText(Math.round(s), x(s), h - bottom + 6);
  ctx.beginPath();
  state.losses.forEach((v, i) => ctx.lineTo(x(i + 1), y(v)));
  ctx.strokeStyle = COLOR.blue;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

worker.onmessage = ({ data: m }) => {
  if (m.type === 'ready') {
    state.ready = true;
    render();
    requestTrace();
    if (generateVisible) generate.load();
  }
  if (m.type === 'modelReady') generate.modelReady();
  if (m.type === 'token') generate.token(m);
  if (m.type === 'generated') generate.generated();
  if (m.type === 'step') onStep(m);
  if (m.type === 'trace') onTrace(m);
  if (m.type === 'sample') onSample(m);
};

$('train-btn').onclick = () => (state.running ? pause() : start());

recorded.then(async rec => {
  state.info = rec.info;
  const buf = new Float32Array(await fetch('recorded/trace0.bin').then(r => r.arrayBuffer()));
  onTrace({ step: 0, ...rec.trace, buf });
  onSample({ step: 0, text: 'Press Train. The model starts from random weights.' });
  render();
  let last = performance.now();
  requestAnimationFrame(function frame(now) {
    const dt = Math.min(50, now - last);
    last = now;
    micro.tick(dt, state.running);
    micro.draw();
    if (state.running || redraw-- > 0) {
      arch(state, dt);
      block.draw(state.info.B, state.info.T, state.sel.row);
      drawLoss();
    }
    requestAnimationFrame(frame);
  });
});
