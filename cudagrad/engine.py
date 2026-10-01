from cudagrad.backend import np, K


_grad_enabled = True


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *_):
        global _grad_enabled
        _grad_enabled = self._prev


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Matrix:

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children) if _grad_enabled else set()
        self._op = _op if _grad_enabled else ''

    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.data + other.data, (self, other), '+')
        if not _grad_enabled:
            return out

        def _backward():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Matrix(other * self.data, (self,), '*')
            if not _grad_enabled:
                return out
            def _backward():
                self.grad += other * out.grad
            out._backward = _backward
            return out

        out = Matrix(self.data * other.data, (self, other), '*')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(K.matmul(self.data, other.data), (self, other), '@')
        if not _grad_enabled:
            return out

        def _backward():
            self.grad  += _unbroadcast(K.matmul(out.grad, other.data.swapaxes(-1, -2)), self.data.shape)
            other.grad += _unbroadcast(K.matmul(self.data.swapaxes(-1, -2), out.grad), other.data.shape)
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Matrix(K.pow_scalar(self.data, other), (self,), f'**{other}')
        if not _grad_enabled:
            return out

        def _backward():
            self.grad += (other * np.power(self.data, other - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Matrix(K.relu(self.data), (self,), 'ReLU')
        if not _grad_enabled:
            return out

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None):
        keepdims = axis is not None
        out = Matrix(np.asarray(self.data.sum(axis=axis, keepdims=keepdims)), (self,), 'sum')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out

    def mean(self, axis=None):
        keepdims = axis is not None
        out = Matrix(np.asarray(self.data.mean(axis=axis, keepdims=keepdims)), (self,), 'mean')
        if not _grad_enabled:
            return out
        n = self.data.size if axis is None else self.data.shape[axis]
        def _backward():
            self.grad += out.grad / n
        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        loss_data, grad_z = K.cross_entropy(self.data, targets)
        out = Matrix(loss_data, (self,), 'cross_entropy')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += grad_z * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Matrix(K.exp(self.data), (self,), 'exp')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Matrix(K.log(self.data), (self,), 'log')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) and self.data.ndim == 2 and idx.dtype.kind in 'iu':
            out = Matrix(K.embedding_gather(self.data, idx), (self,), 'getitem')
            if not _grad_enabled:
                return out
            def _backward():
                K.embedding_scatter_add(self.grad, idx, out.grad)
            out._backward = _backward
            return out

        # np.add.at so repeated indices accumulate instead of overwriting
        out = Matrix(self.data[idx], (self,), 'getitem')
        if not _grad_enabled:
            return out
        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x2 = self.data.reshape(-1, self.data.shape[-1])
        y, mean, rstd = K.layernorm_fwd(x2, gamma.data, beta.data, eps)
        out = Matrix(y.reshape(self.data.shape), (self, gamma, beta), 'layernorm')
        if not _grad_enabled:
            return out
        def _backward():
            dx, dgamma, dbeta = K.layernorm_bwd(out.grad.reshape(x2.shape), x2, gamma.data, mean, rstd)
            self.grad += dx.reshape(self.data.shape)
            gamma.grad += dgamma
            beta.grad += dbeta
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Matrix(self.data.reshape(*shape), (self,), 'reshape')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def swapaxes(self, axis1, axis2):
        out = Matrix(self.data.swapaxes(axis1, axis2), (self,), 'swapaxes')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += out.grad.swapaxes(axis1, axis2)
        out._backward = _backward
        return out

    def dropout(self, p, training):
        if not training or p == 0:
            return self
        mask = (np.random.rand(*self.data.shape) >= p) / (1 - p)
        out = Matrix(self.data * mask, (self,), 'dropout')
        if not _grad_enabled:
            return out
        def _backward():
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.swapaxes(-1, -2)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Matrix(data={self.data}, grad={self.grad})"
