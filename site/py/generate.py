"""The Generate model: the pretrained GPT, sampled one token at a time."""
import json

import numpy as np

from matrixgrad.engine import no_grad
from matrixgrad.nn import GPT

TEMPERATURE, TOP_K, SHOWN = 0.8, 50, 5


class Generator:
    def __init__(self, weights, meta):
        c = json.loads(meta)['config']
        self.model = GPT(c['vocab_size'], c['n_embd'], c['n_head'], c['n_layer'], c['block_size'], 0.0, tie_weights=True)
        flat, off = np.frombuffer(weights, np.float16).astype(np.float32), 0
        for p in self.model.parameters():
            p.data = flat[off:off + p.data.size].reshape(p.data.shape)
            off += p.data.size
        assert off == flat.size
        self.ids = []

    def start(self, ids):
        self.ids = list(ids)

    def next(self):
        """Samples one token; returns it with the SHOWN most likely candidates."""
        with no_grad():
            logits = self.logits(self.ids[-self.model.block_size:]) / TEMPERATURE
        logits[logits < np.partition(logits, -TOP_K)[-TOP_K]] = -np.inf
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        token = int(np.random.choice(len(probs), p=probs))
        self.ids.append(token)
        top = np.argsort(probs)[::-1][:SHOWN]
        return json.dumps({'token': token, 'top': [[int(i), float(probs[i])] for i in top]})

    def logits(self, ids):
        """Next-token logits. Recomputes the whole context each call; a KV cache replaces this."""
        m = self.model
        x = m.wte(np.array([ids])) + m.wpe(np.arange(len(ids)))
        for block in m.blocks:
            x = block(x)
        return m.ln_f(x).data[0, -1] @ m.wte.weight.data.T + m.lm_head_bias.data
