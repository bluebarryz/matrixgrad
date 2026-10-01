import cupy as cp
import pytest
from cudagrad.kernels.bindings import softmax_row_fwd, softmax_row_bwd


def _softmax_ref(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = cp.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (1, 4096), (33, 257)])
def test_softmax_fwd(shape):
    z = cp.random.randn(*shape, dtype=cp.float32) * 5  # exercise the max-stability trick
    y = softmax_row_fwd(z)
    assert cp.allclose(y, _softmax_ref(z), rtol=1e-4, atol=1e-5)
    assert cp.allclose(y.sum(axis=-1), cp.ones(shape[0], dtype=cp.float32), atol=1e-5)


@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (33, 257)])
def test_softmax_bwd_matches_analytic(shape):
    z = cp.random.randn(*shape, dtype=cp.float32)
    s = _softmax_ref(z)
    dy = cp.random.randn(*shape, dtype=cp.float32)
    dz_kernel = softmax_row_bwd(s, dy)
    dot = (dy * s).sum(axis=-1, keepdims=True)
    dz_ref = s * (dy - dot)
    assert cp.allclose(dz_kernel, dz_ref, rtol=1e-4, atol=1e-5)


def test_softmax_finite_difference():
    cp.random.seed(0)
    N, D = 4, 8
    z = cp.random.randn(N, D, dtype=cp.float32)
    s = softmax_row_fwd(z)
    w = cp.random.randn(N, D, dtype=cp.float32)
    dz_analytic = softmax_row_bwd(s, w)

    eps = 1e-3
    dz_fd = cp.zeros_like(z)
    for i in range(N):
        for j in range(D):
            z_pos = z.copy(); z_pos[i, j] += eps
            z_neg = z.copy(); z_neg[i, j] -= eps
            L_pos = (w * softmax_row_fwd(z_pos)).sum()
            L_neg = (w * softmax_row_fwd(z_neg)).sum()
            dz_fd[i, j] = (L_pos - L_neg) / (2 * eps)
    assert cp.allclose(dz_analytic, dz_fd, rtol=1e-2, atol=1e-3)
