import cupy as cp
import pytest
from cudagrad.kernels.bindings import layernorm_fwd, layernorm_bwd


def _layernorm_ref(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y_hat = (x - mean) / cp.sqrt(var + eps)
    return y_hat * gamma + beta, mean[:, 0], 1.0 / cp.sqrt(var + eps)[:, 0]


@pytest.mark.parametrize("shape", [(4, 8), (32, 64), (16, 768)])
def test_layernorm_fwd(shape):
    x = cp.random.randn(*shape, dtype=cp.float32)
    gamma = cp.random.randn(shape[1], dtype=cp.float32)
    beta = cp.random.randn(shape[1], dtype=cp.float32)
    y, mean, rstd = layernorm_fwd(x, gamma, beta, 1e-5)
    y_ref, mean_ref, rstd_ref = _layernorm_ref(x, gamma, beta, 1e-5)
    assert cp.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    assert cp.allclose(mean, mean_ref, atol=1e-5)
    assert cp.allclose(rstd, rstd_ref, rtol=1e-4)


@pytest.mark.parametrize("shape", [(8, 16), (32, 64)])
def test_layernorm_bwd_matches_finite_diff(shape):
    cp.random.seed(0)
    x = cp.random.randn(*shape, dtype=cp.float32)
    gamma = cp.random.randn(shape[1], dtype=cp.float32)
    beta = cp.random.randn(shape[1], dtype=cp.float32)
    eps = 1e-5
    y, mean, rstd = layernorm_fwd(x, gamma, beta, eps)
    dy = cp.random.randn(*shape, dtype=cp.float32)
    dx, dgamma, dbeta = layernorm_bwd(dy, x, gamma, mean, rstd)

    h = 1e-3
    for (i, j) in [(0, 0), (1, 3), (shape[0] - 1, shape[1] - 1)]:
        xp = x.copy(); xp[i, j] += h
        xn = x.copy(); xn[i, j] -= h
        y_pos, _, _ = layernorm_fwd(xp, gamma, beta, eps)
        y_neg, _, _ = layernorm_fwd(xn, gamma, beta, eps)
        fd = float((dy * (y_pos - y_neg)).sum()) / (2 * h)
        assert abs(float(dx[i, j]) - fd) < 1e-1
