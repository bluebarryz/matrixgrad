"""The Train model: a character-level GPT on tiny Shakespeare, trained by matrixgrad."""
import json
from contextlib import nullcontext

import numpy as np

from matrixgrad.nn import GPT, gpt2_init_
from matrixgrad.optimizer import AdamW, clip_grad_norm_
from recorder import grad_norms, recording

B, T = 8, 64
LR, WARMUP = 3e-3, 50


class Trainer:
    def __init__(self, text, seed=0):
        np.random.seed(seed)
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        self.model = GPT(len(self.chars), 64, 4, 2, T, 0.0, tie_weights=True)
        gpt2_init_(self.model)
        self.params = self.model.parameters()
        matrices = [p for p in self.params if p.data.ndim >= 2]
        vectors = [p for p in self.params if p.data.ndim < 2]
        self.opts = [AdamW(matrices, lr=LR, betas=(0.9, 0.99), weight_decay=0.1), AdamW(vectors, lr=LR, betas=(0.9, 0.99))]
        self.step = 0
        self.rec = self.buf = None

    def info(self):
        return json.dumps({'vocab': ''.join(self.chars), 'params': sum(p.data.size for p in self.params), 'B': B, 'T': T})

    def train_step(self, trace=None):
        """One AdamW step. `trace` is (component, row, head) to also record that component's ops."""
        ix = np.random.randint(0, len(self.data) - T - 1, B)
        for opt in self.opts:
            opt.lr = LR * min(1, (self.step + 1) / WARMUP)
        out = self._forward_backward(ix, trace)
        for opt in self.opts:
            opt.step()
        if trace:
            _, row, head = trace
            self.buf = np.concatenate([self.buf, self.rec.after(row, head, B, T)])
            out['trace']['after'] = True
        self.step += 1
        return json.dumps(out | {'step': self.step, 'ix': ix.tolist()})

    def trace_now(self, trace):
        """Traces a fresh batch on the current weights, without training."""
        ix = np.random.randint(0, len(self.data) - T - 1, B)
        return json.dumps(self._forward_backward(ix, trace))

    def trace_at(self, weights, ix, trace):
        """Traces one batch on snapshot weights, without training."""
        self.load(weights, np.float16, optimizer=False)
        return json.dumps(self._forward_backward(np.asarray(ix), trace))

    def _forward_backward(self, ix, trace):
        windows = np.stack([self.data[i:i + T + 1] for i in ix])
        for opt in self.opts:
            opt.zero_grad()
        with recording(self.model) if trace else nullcontext() as rec:
            _, loss = self.model(windows[:, :-1], windows[:, 1:])
        loss.backward()
        clip_grad_norm_(self.params, 1.0)

        out = {'loss': float(loss.data), 'norms': grad_norms(self.model)}
        if trace:
            component, row, head = trace
            self.rec = rec
            out['trace'], self.buf = rec.export(component, row, head, B, T)
            out['trace'] |= {'component': component, 'row': row, 'head': head,
                             'batch': [''.join(self.chars[i] for i in w) for w in windows]}
        return out

    def _arrays(self, optimizer):
        return [p.data for p in self.params] + ([a for opt in self.opts for a in opt.m + opt.v] if optimizer else [])

    def save(self, dtype=np.float32, optimizer=True):
        return np.concatenate([a.ravel() for a in self._arrays(optimizer)]).astype(dtype)

    def load(self, buf, dtype=np.float32, optimizer=True, step=None):
        flat = np.frombuffer(buf, dtype).astype(np.float32)
        off = 0
        for a in self._arrays(optimizer):
            a[...] = flat[off:off + a.size].reshape(a.shape)
            off += a.size
        assert off == flat.size
        if optimizer:
            self.step = step
            for opt in self.opts:
                opt.t = step

    def sample(self, n=160):
        out = self.model.generate(np.array([[self.stoi['\n']]]), n, temperature=0.8)
        return ''.join(self.chars[i] for i in out[0, 1:])
