// Generate: the pretrained GPT, streamed token by token with its top candidates.
const TIKTOKEN = 'https://cdn.jsdelivr.net/npm/js-tiktoken@1.0.21';
const TOKENS = 200;

const span = (cls, text) => Object.assign(document.createElement('span'), { className: cls, textContent: text });
const mb = n => `${Math.round(n / 1e6)} MB`;

export function createGenerate(el, worker, { onStart, onEnd }) {
  let enc, loading = false, running = false, times = [], partial = [];

  async function load() {
    if (loading) return;
    loading = true;
    const res = await fetch('model/gpt.json');
    if (!res.ok) return el.out.replaceChildren(span('wait', 'The pretrained model is coming soon.'));
    el.out.replaceChildren(span('wait', 'loading model…'));
    const [{ Tiktoken }, { default: ranks }, meta] = await Promise.all([
      import(`${TIKTOKEN}/lite/+esm`),
      import(`${TIKTOKEN}/dist/ranks/gpt2.js/+esm`),
      res.json(),
    ]);
    enc = new Tiktoken(ranks);
    const weights = new Uint8Array(meta.bytes), reader = (await fetch('model/gpt.bin')).body.getReader();
    for (let got = 0, chunk; !(chunk = await reader.read()).done; got += chunk.value.length) {
      weights.set(chunk.value, got);
      el.out.firstChild.textContent = `loading model · ${mb(got)} / ${mb(meta.bytes)}`;
    }
    worker.postMessage({ type: 'loadModel', weights: weights.buffer, meta: JSON.stringify(meta) }, [weights.buffer]);
  }

  function bars({ token, top }) {
    el.bars.replaceChildren(span('cap', 'next-token probabilities'), ...top.map(([id, p]) => {
      const row = document.createElement('div');
      row.className = id === token ? 'pbar pick' : 'pbar';
      const fill = span('fill', '');
      fill.style.width = `${(p * 100).toFixed(0)}%`;
      row.append(span('w', JSON.stringify(enc.decode([id])).slice(1, -1)), fill, span('n', p.toFixed(2)));
      return row;
    }));
  }

  el.form.onsubmit = e => {
    e.preventDefault();
    if (running) return worker.postMessage({ type: 'stop' });
    running = true;
    onStart();
    times = [];
    partial = [];
    el.button.textContent = 'Stop';
    el.out.replaceChildren(span('p', el.input.value));
    worker.postMessage({ type: 'generate', ids: enc.encode(el.input.value), n: TOKENS });
  };

  return {
    load,
    modelReady() {
      el.out.replaceChildren(span('wait', 'model ready'));
      el.button.disabled = false;
    },
    token(m) {
      // byte-level BPE can split one character across tokens; hold them until it decodes whole
      partial.push(m.token);
      const text = enc.decode(partial);
      if (!text.endsWith('\uFFFD')) {
        el.out.append(span('t', text));
        partial = [];
      }
      bars(m);
      times.push(m.ms);
      el.stat.textContent = `${Math.round(times.reduce((a, b) => a + b) / times.length)} ms/token · KV cache`;
    },
    generated() {
      running = false;
      el.button.textContent = 'Generate';
      onEnd();
    },
  };
}
