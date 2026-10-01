try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


class _Kernels:
    USE_CUSTOM_MATMUL = False
    USE_CUSTOM_ELEMENTWISE = False
    USE_CUSTOM_LAYERNORM = False
    USE_CUSTOM_CROSS_ENTROPY = False
    USE_CUSTOM_EMBEDDING = False
    USE_CUSTOM_ADAMW = False

    def matmul(self, a, b):
        if self.USE_CUSTOM_MATMUL and GPU:
            from cudagrad.kernels.bindings import sgemm
            return sgemm(a, b)
        return a @ b

    def exp(self, x):
        if self.USE_CUSTOM_ELEMENTWISE and GPU:
            from cudagrad.kernels.bindings import elementwise_unary
            return elementwise_unary(x, op="exp")
        return np.exp(x)

    def log(self, x):
        if self.USE_CUSTOM_ELEMENTWISE and GPU:
            from cudagrad.kernels.bindings import elementwise_unary
            return elementwise_unary(x, op="log")
        return np.log(x)

    def relu(self, x):
        if self.USE_CUSTOM_ELEMENTWISE and GPU:
            from cudagrad.kernels.bindings import elementwise_unary
            return elementwise_unary(x, op="relu")
        return np.maximum(0, x)

    def pow_scalar(self, x, p):
        if self.USE_CUSTOM_ELEMENTWISE and GPU:
            from cudagrad.kernels.bindings import elementwise_pow
            return elementwise_pow(x, float(p))
        return np.power(x, p)

    def layernorm_fwd(self, x, gamma, beta, eps):
        if self.USE_CUSTOM_LAYERNORM and GPU and x.ndim == 2:
            from cudagrad.kernels.bindings import layernorm_fwd
            return layernorm_fwd(x, gamma, beta, eps)
        mean = x.mean(axis=-1)
        centered = x - mean[:, None]
        var = (centered * centered).mean(axis=-1)
        rstd = 1.0 / np.sqrt(var + eps)
        y = centered * rstd[:, None] * gamma + beta
        return y, mean, rstd

    def layernorm_bwd(self, dy, x, gamma, mean, rstd):
        if self.USE_CUSTOM_LAYERNORM and GPU and x.ndim == 2:
            from cudagrad.kernels.bindings import layernorm_bwd
            return layernorm_bwd(dy, x, gamma, mean, rstd)
        y_hat = (x - mean[:, None]) * rstd[:, None]
        dh = dy * gamma
        a = dh.mean(axis=-1, keepdims=True)
        b = (dh * y_hat).mean(axis=-1, keepdims=True)
        dx = rstd[:, None] * (dh - a - y_hat * b)
        dgamma = (dy * y_hat).sum(axis=0)
        dbeta = dy.sum(axis=0)
        return dx, dgamma, dbeta

    def cross_entropy(self, z, targets):
        if self.USE_CUSTOM_CROSS_ENTROPY and GPU and z.ndim == 2:
            from cudagrad.kernels.bindings import cross_entropy_fused
            return cross_entropy_fused(z, targets)
        N = z.shape[0]
        zs = z - z.max(axis=1, keepdims=True)
        e = np.exp(zs)
        s = e.sum(axis=1, keepdims=True)
        nll = (np.log(s)[:, 0] - zs[np.arange(N), targets]).mean()
        gz = e / s
        gz[np.arange(N), targets] -= 1.0
        gz /= N
        return np.array(nll), gz

    def embedding_gather(self, table, idx):
        if self.USE_CUSTOM_EMBEDDING and GPU and table.ndim == 2:
            from cudagrad.kernels.bindings import embedding_gather
            return embedding_gather(table, idx)
        return table[idx]

    def embedding_scatter_add(self, dst, idx, src):
        if self.USE_CUSTOM_EMBEDDING and GPU and dst.ndim == 2:
            from cudagrad.kernels.bindings import embedding_scatter_add
            embedding_scatter_add(dst, idx, src)
            return
        np.add.at(dst, idx, src)

    def adamw_step(self, p, g, m, v, lr, b1, b2, eps, bc1, bc2, wd):
        if self.USE_CUSTOM_ADAMW and GPU:
            from cudagrad.kernels.bindings import adamw_step
            adamw_step(p, g, m, v, lr, b1, b2, eps, bc1, bc2, wd)
            return
        if wd != 0.0:
            p -= lr * wd * p
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        p -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)


K = _Kernels()
