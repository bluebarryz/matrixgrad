from cudagrad.backend import np, K


class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class SGD(Optimizer):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.b1, self.b2 = betas
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p.data -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        bc1 = float(1 - self.b1 ** self.t)
        bc2 = float(1 - self.b2 ** self.t)
        for i, p in enumerate(self.params):
            K.adamw_step(p.data, p.grad, self.m[i], self.v[i], self.lr, self.b1, self.b2,
                         self.eps, bc1, bc2, self.weight_decay)


def clip_grad_norm_(params, max_norm, eps=1e-6):
    total_sq = 0.0
    for p in params:
        total_sq += float((p.grad * p.grad).sum())
    total_norm = total_sq ** 0.5
    coef = max_norm / (total_norm + eps)
    if coef < 1.0:
        for p in params:
            p.grad *= coef
    return total_norm
