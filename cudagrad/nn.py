from cudagrad.engine import Matrix, no_grad
from cudagrad.backend import np


class Module:
    training = True

    def train(self, mode=True):
        self.training = mode
        for child in self._children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _children(self):
        for v in vars(self).values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Module):
                        yield item

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, nin, nout):
        self.W = Matrix(np.random.randn(nin, nout) / np.sqrt(nin))
        self.b = Matrix(np.zeros(nout))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class Dropout(Module):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return x.dropout(self.p, self.training)


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(nouts) - 1:
                self.layers.append(ReLU())

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class CausalSelfAttention(Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = Linear(n_embd, 3 * n_embd)
        self.c_proj = Linear(n_embd, n_embd)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        self.mask = np.triu(np.full((block_size, block_size), -np.inf), k=1) \
                      .reshape(1, 1, block_size, block_size)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q = qkv[:, :, 0].swapaxes(1, 2)   # (B, n_head, T, head_dim)
        k = qkv[:, :, 1].swapaxes(1, 2)
        v = qkv[:, :, 2].swapaxes(1, 2)

        scores = (q @ k.swapaxes(-1, -2)) * (self.head_dim ** -0.5)
        scores = scores + self.mask[:, :, :T, :T]

        # subtract the row max so exp can't overflow
        scores.data -= scores.data.max(axis=-1, keepdims=True)
        e = scores.exp()
        attn = self.attn_dropout(e / e.sum(axis=-1))

        y = (attn @ v).swapaxes(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def parameters(self):
        return self.c_attn.parameters() + self.c_proj.parameters()


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Matrix(np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def __call__(self, idx):
        return self.weight[idx]

    def parameters(self):
        return [self.weight]


class LayerNorm(Module):
    def __init__(self, n_embd, eps=1e-5):
        self.gamma = Matrix(np.ones(n_embd))
        self.beta = Matrix(np.zeros(n_embd))
        self.eps = eps

    def __call__(self, x):
        return x.layernorm(self.gamma, self.beta, self.eps)

    def parameters(self):
        return [self.gamma, self.beta]


class FeedForward(Module):
    def __init__(self, n_embd, dropout):
        self.c_fc = Linear(n_embd, 4 * n_embd)
        self.c_proj = Linear(4 * n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        return self.dropout(self.c_proj(self.c_fc(x).relu()))

    def parameters(self):
        return self.c_fc.parameters() + self.c_proj.parameters()


class Block(Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def parameters(self):
        return (self.ln_1.parameters() + self.attn.parameters()
                + self.ln_2.parameters() + self.mlp.parameters())


class GPT(Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size,
                 dropout, tie_weights=False):
        self.block_size = block_size
        self.tie_weights = tie_weights
        self.wte = Embedding(vocab_size, n_embd)
        self.wpe = Embedding(block_size, n_embd)
        self.drop = Dropout(dropout)
        self.blocks = [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        if tie_weights:
            self.lm_head_bias = Matrix(np.zeros(vocab_size))
        else:
            self.lm_head = Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(np.arange(T))
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.tie_weights:
            logits = x @ self.wte.weight.T + self.lm_head_bias
        else:
            logits = self.lm_head(x)

        if targets is None:
            return logits, None
        V = logits.shape[-1]
        loss = logits.reshape(B * T, V).cross_entropy(targets.reshape(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        idx = np.asarray(idx)
        with no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:] if idx.shape[1] > self.block_size else idx
                logits, _ = self(idx_cond)
                logits = logits.data[:, -1, :] / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    kth = np.partition(logits, -k, axis=-1)[:, -k]
                    logits = np.where(logits < kth[:, None], -np.inf, logits)
                logits -= logits.max(axis=-1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=-1, keepdims=True)
                # cupy's random.choice needs an explicit size
                idx_next = np.array([np.random.choice(probs.shape[-1], size=1, p=row)[0] for row in probs])
                idx = np.concatenate([idx, idx_next[:, None]], axis=1)
        return idx

    def parameters(self):
        params = self.wte.parameters() + self.wpe.parameters() + self.ln_f.parameters()
        if self.tie_weights:
            params.append(self.lm_head_bias)
        else:
            params += self.lm_head.parameters()
        for block in self.blocks:
            params += block.parameters()
        seen, out = set(), []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out


def gpt2_init_(model):
    n_layer = len(model.blocks)
    std_resid = 0.02 / np.sqrt(2 * n_layer)

    def _normal(shape, std):
        return (np.random.randn(*shape) * std).astype(np.float32)

    model.wte.weight.data = _normal(model.wte.weight.shape, 0.02)
    model.wpe.weight.data = _normal(model.wpe.weight.shape, 0.02)
    if model.tie_weights:
        model.lm_head_bias.data = np.zeros_like(model.lm_head_bias.data)
    else:
        model.lm_head.W.data = _normal(model.lm_head.W.shape, 0.02)
        model.lm_head.b.data = np.zeros_like(model.lm_head.b.data)

    for blk in model.blocks:
        blk.attn.c_attn.W.data = _normal(blk.attn.c_attn.W.shape, 0.02)
        blk.attn.c_attn.b.data = np.zeros_like(blk.attn.c_attn.b.data)
        blk.attn.c_proj.W.data = _normal(blk.attn.c_proj.W.shape, std_resid)
        blk.attn.c_proj.b.data = np.zeros_like(blk.attn.c_proj.b.data)
        blk.mlp.c_fc.W.data    = _normal(blk.mlp.c_fc.W.shape, 0.02)
        blk.mlp.c_fc.b.data    = np.zeros_like(blk.mlp.c_fc.b.data)
        blk.mlp.c_proj.W.data  = _normal(blk.mlp.c_proj.W.shape, std_resid)
        blk.mlp.c_proj.b.data  = np.zeros_like(blk.mlp.c_proj.b.data)
