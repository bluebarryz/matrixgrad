importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');

const SOURCES = [
  'matrixgrad/__init__.py', 'matrixgrad/backend.py', 'matrixgrad/engine.py', 'matrixgrad/nn.py', 'matrixgrad/optimizer.py',
  'py/recorder.py', 'py/train.py',
];
const SAMPLE_EVERY = 100;

let py, trainer, running = false, traceRequest = null;

async function boot() {
  py = await loadPyodide();
  const [, files, text] = await Promise.all([
    py.loadPackage('numpy'),
    Promise.all(SOURCES.map(f => fetch(f).then(r => r.text()))),
    fetch('data/shakespeare.txt').then(r => r.text()),
  ]);
  py.FS.mkdirTree('/home/pyodide/matrixgrad');
  SOURCES.forEach((f, i) => py.FS.writeFile('/home/pyodide/' + f.replace('py/', ''), files[i]));
  py.globals.set('text', text);
  trainer = py.runPython('from train import Trainer; Trainer(text)');
  postMessage({ type: 'ready', ...JSON.parse(trainer.info()) });
}

function postTrace(step, trace) {
  const proxy = trainer.buf, view = proxy.getBuffer('f32');
  const buf = view.data.slice();
  view.release();
  proxy.destroy();
  postMessage({ type: 'trace', step, ...trace, buf }, [buf.buffer]);
}

async function loop() {
  while (running) {
    const req = traceRequest;
    traceRequest = null;
    const t0 = performance.now();
    const { trace, ...step } = JSON.parse(trainer.train_step(req && py.toPy([req.component, req.row, req.head])));
    postMessage({ type: 'step', ...step, ms: performance.now() - t0 });
    if (trace) postTrace(step.step, trace);
    if (step.step % SAMPLE_EVERY === 0) postMessage({ type: 'sample', step: step.step, text: trainer.sample() });
    await new Promise(r => setTimeout(r));
  }
}

const actions = {
  run() {
    if (running) return;
    running = true;
    loop();
  },
  pause() { running = false; },
  trace(m) { traceRequest = m; },
  load(m) { trainer.load(py.toPy(new Uint8Array(m.buf)), 'float32', true, m.step); },
  traceNow(m) {
    const out = trainer.trace_now(py.toPy([m.component, m.row, m.head]));
    postTrace(trainer.step, JSON.parse(out).trace);
  },
  traceAt(m) {
    const out = trainer.trace_at(py.toPy(new Uint8Array(m.weights)), py.toPy(m.ix), py.toPy([m.component, m.row, m.head]));
    postTrace(m.step, JSON.parse(out).trace);
  },
};

onmessage = ({ data }) => actions[data.type](data);

boot();
