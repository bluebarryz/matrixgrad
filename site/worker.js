importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');

const SOURCES = [
  'matrixgrad/__init__.py', 'matrixgrad/backend.py', 'matrixgrad/engine.py', 'matrixgrad/nn.py', 'matrixgrad/optimizer.py',
  'py/recorder.py', 'py/train.py', 'py/generate.py',
];
const SAMPLE_EVERY = 100;

let py, trainer, generator, running = false, generating = false, traceRequest = null;

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

async function generate(n) {
  generating = true;
  for (let i = 0; i < n && generating; i++) {
    const t0 = performance.now();
    const token = JSON.parse(generator.next());
    postMessage({ type: 'token', ...token, ms: performance.now() - t0 });
    await new Promise(r => setTimeout(r));
  }
  generating = false;
  postMessage({ type: 'generated' });
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
  loadModel(m) {
    generator = py.runPython('from generate import Generator; Generator')(py.toPy(new Uint8Array(m.weights)), m.meta);
    postMessage({ type: 'modelReady' });
  },
  generate(m) {
    generator.start(py.toPy(m.ids));
    generate(m.n);
  },
  stop() { generating = false; },
  traceAt(m) {
    const out = trainer.trace_at(py.toPy(new Uint8Array(m.weights)), py.toPy(m.ix), py.toPy([m.component, m.row, m.head]));
    postTrace(m.step, JSON.parse(out).trace);
  },
};

onmessage = ({ data }) => actions[data.type](data);

boot();
