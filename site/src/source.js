// What each component is, and the matrixgrad source behind each op it records.

export const COMPONENTS = {
  emb: ['embedding', 'wte + wpe · 8,256 params'],
  'b0.ln1': ['block 0 · ln_1', 'LayerNorm · 128 params'],
  'b0.attn': ['block 0 · attention', 'CausalSelfAttention, 4 heads · 16,640 params'],
  'b0.ln2': ['block 0 · ln_2', 'LayerNorm · 128 params'],
  'b0.mlp': ['block 0 · MLP', 'FeedForward · 33,088 params'],
  'b1.ln1': ['block 1 · ln_1', 'LayerNorm · 128 params'],
  'b1.attn': ['block 1 · attention', 'CausalSelfAttention, 4 heads · 16,640 params'],
  'b1.ln2': ['block 1 · ln_2', 'LayerNorm · 128 params'],
  'b1.mlp': ['block 1 · MLP', 'FeedForward · 33,088 params'],
  lnf: ['ln_f', 'LayerNorm · 128 params'],
  head: ['head', 'tied to wte · 65 params'],
  loss: ['loss', 'cross-entropy over 512 positions'],
};

export const typeOf = c => c === 'lnf' ? 'ln' : c.includes('.') ? c.split('.')[1].replace(/\d$/, '') : c;

export const FORWARD = {
  emb: {
    fn: 'matrixgrad/nn.py · GPT.__call__',
    lines: ['tok_emb = self.wte(idx)', 'pos_emb = self.wpe(np.arange(T))', 'x = self.drop(tok_emb + pos_emb)'],
    ops: [[0, 'tok_emb'], [1, 'pos_emb'], [2, 'x']],
  },
  ln: {
    fn: 'matrixgrad/nn.py · LayerNorm.__call__',
    lines: ['def __call__(self, x):', '    mean = x.mean(axis=-1)', '    centered = x - mean', '    var = (centered * centered).mean(axis=-1)',
      '    scale = (var + self.eps) ** -0.5', '    return centered * scale * self.gamma + self.beta'],
    ops: [[1, 'mean'], [2, '−mean'], [2, 'centered'], [3, 'centered²'], [3, 'var'], [4, 'var + eps'], [4, 'scale'],
      [5, 'normed'], [5, '· gamma'], [5, 'out']],
  },
  attn: {
    fn: 'matrixgrad/nn.py · CausalSelfAttention.__call__',
    lines: ['def __call__(self, x):', '    qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, self.head_dim)',
      '    q = qkv[:, :, 0].swapaxes(1, 2)   # (B, n_head, T, head_dim)', '    k = qkv[:, :, 1].swapaxes(1, 2)',
      '    v = qkv[:, :, 2].swapaxes(1, 2)', '    scores = (q @ k.swapaxes(-1, -2)) * (self.head_dim ** -0.5)',
      '    scores = scores + self.mask[:, :, :T, :T]', '    scores.data -= scores.data.max(axis=-1, keepdims=True)',
      '    e = scores.exp()', '    attn = self.attn_dropout(e / e.sum(axis=-1))', '    y = (attn @ v).swapaxes(1, 2).reshape(B, T, C)',
      '    return self.resid_dropout(self.c_proj(y))'],
    ops: [[1, 'x @ W'], [1, 'qkv'], [1, 'qkv'], [2, 'q'], [2, 'q'], [3, 'k'], [3, 'k'], [4, 'v'], [4, 'v'], [5, 'kᵀ'],
      [5, 'q kᵀ'], [5, 'scores'], [6, 'masked'], [8, 'e'], [9, 'Σe'], [9, '1/Σe'], [9, 'attn'], [10, 'attn @ v'], [10, 'y'],
      [10, 'y'], [11, 'y @ W'], [11, 'out']],
  },
  mlp: {
    fn: 'matrixgrad/nn.py · FeedForward.__call__, Linear.__call__',
    lines: ['class FeedForward(Module):', '    def __call__(self, x):', '        return self.dropout(self.c_proj(self.c_fc(x).relu()))', '',
      'class Linear(Module):', '    def __call__(self, x):', '        return x @ self.W + self.b'],
    ops: [[2, 'x @ W_fc'], [2, 'h'], [2, 'relu(h)'], [2, '· W_proj'], [2, 'out']],
  },
  head: {
    fn: 'matrixgrad/nn.py · GPT.__call__',
    lines: ['x = self.ln_f(x)', 'if self.tie_weights:', '    logits = x @ self.wte.weight.T + self.lm_head_bias'],
    ops: [[2, 'wteᵀ'], [2, 'x @ wteᵀ'], [2, 'logits']],
  },
  loss: {
    fn: 'matrixgrad/nn.py · GPT.__call__',
    lines: ['V = logits.shape[-1]', 'loss = logits.reshape(B * T, V).cross_entropy(targets.reshape(-1))'],
    ops: [[1, 'logits'], [1, 'loss']],
  },
};

const pair = (a, b) => [`self.grad  += _unbroadcast(${a}, self.data.shape)`, `other.grad += _unbroadcast(${b}, other.data.shape)`];

export const BACKWARD = {
  '@': ['Matrix.__matmul__', pair('out.grad @ other.data.swapaxes(-1, -2)', 'self.data.swapaxes(-1, -2) @ out.grad')],
  '+': ['Matrix.__add__', pair('out.grad', 'out.grad')],
  '*': ['Matrix.__mul__', pair('other.data * out.grad', 'self.data  * out.grad')],
  '*k': ['Matrix.__mul__', ['self.grad += other * out.grad']],
  '**': ['Matrix.__pow__', ['self.grad += (other * np.power(self.data, other - 1)) * out.grad']],
  ReLU: ['Matrix.relu', ['self.grad += (out.data > 0) * out.grad']],
  sum: ['Matrix.sum', ['self.grad += out.grad']],
  mean: ['Matrix.mean', ['self.grad += out.grad / n']],
  exp: ['Matrix.exp', ['self.grad += out.data * out.grad']],
  getitem: ['Matrix.__getitem__', ['g = np.zeros_like(self.data)', 'np.add.at(g, idx, out.grad)', 'self.grad += g']],
  reshape: ['Matrix.reshape', ['self.grad += out.grad.reshape(self.data.shape)']],
  swapaxes: ['Matrix.swapaxes', ['self.grad += out.grad.swapaxes(axis1, axis2)']],
  cross_entropy: ['Matrix.cross_entropy', ['# dL/dz = (softmax(z) - onehot(targets)) / N', 'self.grad += grad_z * out.grad']],
};

export const backwardKey = op => op.op.startsWith('**') ? '**' : op.op === '*' && op.in.length === 1 ? '*k' : op.op;

export const ADAMW = {
  fn: 'matrixgrad/optimizer.py · AdamW.step',
  lines: ['def step(self):', '    self.t += 1', '    bc1 = 1 - self.b1 ** self.t', '    bc2 = 1 - self.b2 ** self.t',
    '    for i, p in enumerate(self.params):', '        if self.weight_decay != 0:', '            p.data -= self.lr * self.weight_decay * p.data',
    '        g = p.grad', '        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g', '        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)',
    '        p.data -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)'],
  hl: [6, 8, 9, 10],
};
