"""Records what matrixgrad computes in a training step, leaving the engine untouched."""
from contextlib import contextmanager

import numpy as np

from matrixgrad.engine import Matrix
from matrixgrad.nn import Block, CausalSelfAttention, Embedding, FeedForward, LayerNorm, Module

ROWS, COLS = 16, 64
LOSS_OPS = {'reshape', 'cross_entropy'}


def components(model):
    assert model.tie_weights
    comps = {'emb': [model.wte, model.wpe]}
    for i, b in enumerate(model.blocks):
        comps |= {f'b{i}.ln1': [b.ln_1], f'b{i}.attn': [b.attn], f'b{i}.ln2': [b.ln_2], f'b{i}.mlp': [b.mlp]}
    return comps | {'lnf': [model.ln_f]}


def named_params(module, prefix=''):
    for k, v in vars(module).items():
        if isinstance(v, Matrix):
            yield prefix + k, v
        elif isinstance(v, Module):
            yield from named_params(v, f'{prefix}{k}.')
        elif isinstance(v, list):
            for i, m in enumerate(v):
                yield from named_params(m, f'{prefix}{k}.{i}.')


def grad_norms(model):
    groups = {c: [p for m in ms for p in m.parameters()] for c, ms in components(model).items()}
    groups['head'] = [model.lm_head_bias]
    return {c: float(np.sqrt(sum((p.grad ** 2).sum() for p in ps))) for c, ps in groups.items()}


def view(a, row, head, B, T):
    """The slice of `a` the page draws: one batch row, one head, at most ROWS x COLS."""
    if a.ndim >= 3 and a.shape[0] == B:
        a = a[row]
        if a.shape[0] != T:
            a = a[head]
    elif a.ndim == 2 and a.shape[0] == B * T:
        a = a[row * T:]
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    a = a.reshape(len(a), -1) if a.ndim > 1 else a.reshape(1, -1)
    return a[:ROWS, :COLS]


class Recorder:
    def __init__(self, model):
        self.names = {id(m): c for c, ms in components(model).items() for m in ms}
        self.names |= {id(b): 'resid' for b in model.blocks}
        self.params = {id(p): name for name, p in named_params(model)}
        self.tag, self.top = None, 'emb'
        self.ops = []

    def created(self, out, children, op):
        if not op:
            return
        tag = self.tag or ('loss' if op in LOSS_OPS else self.top)
        self.ops.append((tag, out, children))

    def export(self, component, row, head, B, T):
        """One component's ops after backward, as (meta, float32 buffer of value/grad slices)."""
        index, exported, tensors, chunks, ops = {}, [], [], [], []

        def ref(t):
            if id(t) not in index:
                v, g = view(t.data, row, head, B, T), view(t.grad, row, head, B, T)
                index[id(t)] = len(tensors)
                exported.append(t)
                tensors.append({'shape': list(t.data.shape), 'view': list(v.shape), 'param': self.params.get(id(t))})
                chunks.extend((v.ravel(), g.ravel()))
            return index[id(t)]

        for tag, out, children in self.ops:
            if tag == component:
                ops.append({'op': out._op, 'in': [ref(c) for c in children], 'out': ref(out)})
        self.exported = exported
        return {'ops': ops, 'tensors': tensors}, np.concatenate(chunks).astype(np.float32)

    def after(self, row, head, B, T):
        """Slices of the exported parameters as they are now, e.g. after the optimizer step."""
        chunks = [view(t.data, row, head, B, T).ravel() for t in self.exported if id(t) in self.params]
        return np.concatenate([np.zeros(0, np.float32), *chunks]).astype(np.float32)


@contextmanager
def recording(model):
    """Tags every Matrix created inside the block with the component that created it."""
    rec = Recorder(model)
    init = Matrix.__init__
    calls = {cls: cls.__call__ for cls in (Embedding, LayerNorm, CausalSelfAttention, FeedForward, Block)}

    def created(self, data, _children=(), _op=''):
        init(self, data, _children, _op)
        rec.created(self, _children, _op)

    def tagged(cls, call):
        def wrapper(module, *args):
            outer, rec.tag = rec.tag, rec.names[id(module)]
            out = call(module, *args)
            rec.tag = outer
            if cls is Block:
                rec.top = 'head'
            return out
        return wrapper

    Matrix.__init__ = created
    for cls, call in calls.items():
        cls.__call__ = tagged(cls, call)
    yield rec
    Matrix.__init__ = init
    for cls, call in calls.items():
        cls.__call__ = call
