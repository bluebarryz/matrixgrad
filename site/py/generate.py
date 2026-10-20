"""The Generate model: the pretrained GPT, sampled one token at a time."""
import json

import numpy as np

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
        self.steps = None

    def start(self, ids):
        self.steps = self.model.stream(np.array([ids]), TEMPERATURE, TOP_K)

    def next(self):
        """Samples one token; returns it with the SHOWN most likely candidates."""
        token, probs = next(self.steps)
        top = np.argsort(probs[0])[::-1][:SHOWN]
        return json.dumps({'token': int(token[0]), 'top': [[int(i), float(probs[0, i])] for i in top]})
