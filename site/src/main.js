import { createArch } from './arch.js';
import { createBatch, createBlock } from './batch.js';
import { COLOR, decode, fitCanvas } from './heat.js';
import { attentionThumb, createMicro, sameView } from './micro.js';

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const AUTO_PAUSE_TOKENS = 500_000;
const REPLAY_MS = 160;
const HISTORY = 80;
const PHONE = matchMedia('(pointer: coarse) and (max-width: 820px)').matches;

const worker = new Worker('worker.js');
const recorded = fetch('recorded/run.json').then(r => r.json());
let snapshots, redraw = 0;

// engine: booting -> ready -> live. Until it is live, Train replays the Recorded Trace.
const state = {
  info: null, engine: 'booting', running: false, step: 0, replayId: 0,
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

function render() {
  $('train-btn').textContent = state.running ? 'Pause' : state.step ? 'Keep training' : 'Train';
  $('badge').textContent = state.engine === 'live' ? 'live' : 'recorded';
  $('badge').classList.toggle('live', state.engine === 'live');
  redraw = 30;
}

async function start() {
  state.running = true;
  render();
  if (state.engine === 'ready' && state.step === 0 && !PHONE) state.engine = 'live';
  if (state.engine === 'live') return worker.postMessage({ type: 'run' });
  replay(await recorded, ++state.replayId);
}

function pause() {
  state.running = false;
  worker.postMessage({ type: 'pause' });
  render();
}

async function replay(rec, id) {
  const last = rec.handoffs.at(-1);
  while (state.running && id === state.replayId) {
    const s = state.step;
    if (rec.handoffs.includes(s) && state.engine === 'ready' && (!PHONE || s === last)) return handoff(s);
    const [loss, ...norms] = rec.steps[s];
    onStep({ step: s + 1, loss, norms: Object.fromEntries(rec.components.map((c, i) => [c, norms[i]])), ms: REPLAY_MS });
    if (rec.samples[s + 1]) onSample({ step: s + 1, text: rec.samples[s + 1] });
    await sleep(REPLAY_MS);
  }
}

async function handoff(step) {
  const buf = await fetch(`recorded/state${step}.bin`).then(r => r.arrayBuffer());
  worker.postMessage({ type: 'load', buf, step }, [buf]);
  state.engine = 'live';
  if (state.running) worker.postMessage({ type: 'run' });
  render();
}

async function requestTrace() {
  const { component, row, head } = state.sel;
  if (state.engine === 'live') return worker.postMessage({ type: state.running ? 'trace' : 'traceNow', component, row, head });
  if (state.engine === 'booting') return;
  const rec = await recorded;
  snapshots ??= fetch('recorded/snapshots.bin').then(r => r.arrayBuffer());
  const all = await snapshots, size = all.byteLength / rec.snapshots.length;
  const k = rec.snapshots.findLastIndex(s => s.step <= state.step), { step, ix } = rec.snapshots[k];
  const weights = all.slice(k * size, (k + 1) * size);
  worker.postMessage({ type: 'traceAt', weights, ix, step, component, row, head }, [weights]);
}

function select(change) {
  Object.assign(state.sel, change);
  if (state.batch) renderBatch(state.batch, state.sel.row);
  if (state.engine === 'booting') $('m-status').textContent = 'starting the engine…';
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
    recorded.then(rec => showLongRun(rec.samples[Math.max(...Object.keys(rec.samples))]));
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
    if (state.engine === 'booting') state.engine = 'ready';
    if (!state.running) requestTrace();
  }
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
  $('train-btn').disabled = false;
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
