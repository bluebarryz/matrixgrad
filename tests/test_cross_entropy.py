import cupy as cp
import pytest
from cudagrad.kernels.bindings import cross_entropy_fused
from cudagrad.backend import K
from cudagrad.engine import Matrix


def _ref(z, t):
    z = z - z.max(axis=1, keepdims=True)
    e = cp.exp(z)
    s = e.sum(axis=1, keepdims=True)
    log_s = cp.log(s)
    N = z.shape[0]
    nll = (log_s[:, 0] - z[cp.arange(N), t]).mean()
    sm = e / s
    onehot = cp.zeros_like(z)
    onehot[cp.arange(N), t] = 1.0
    grad = (sm - onehot) / N
    return float(nll), grad


@pytest.mark.parametrize("N,C", [(4, 8), (64, 100), (128, 50257)])
def test_cross_entropy_parity(N, C):
    z = cp.random.randn(N, C, dtype=cp.float32) * 3
    t = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    loss, grad_z = cross_entropy_fused(z, t)
    loss_ref, grad_ref = _ref(z, t)
    assert abs(float(loss) - loss_ref) < 1e-3
    assert cp.allclose(grad_z, grad_ref, atol=1e-5, rtol=1e-4)


def test_engine_dispatch_matches_fallback():
    cp.random.seed(0)
    N, C = 32, 16
    z_data = cp.random.randn(N, C, dtype=cp.float32)
    t = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    K.USE_CUSTOM_CROSS_ENTROPY = False
    a = Matrix(z_data.copy())
    la = a.cross_entropy(t); la.backward()

    K.USE_CUSTOM_CROSS_ENTROPY = True
    try:
        b = Matrix(z_data.copy())
        lb = b.cross_entropy(t); lb.backward()
    finally:
        K.USE_CUSTOM_CROSS_ENTROPY = False

    assert abs(float(la.data) - float(lb.data)) < 1e-3
    assert cp.allclose(a.grad, b.grad, rtol=1e-4, atol=1e-5)
