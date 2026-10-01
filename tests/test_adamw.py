import cupy as cp
from cudagrad.kernels.bindings import adamw_step
from cudagrad.backend import K
from cudagrad.engine import Matrix
from cudagrad.optimizer import AdamW


def _ref_step(p, g, m, v, lr, b1, b2, eps, bc1, bc2, wd):
    if wd != 0.0: p -= lr * wd * p
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    p -= lr * (m / bc1) / (cp.sqrt(v / bc2) + eps)
    return p, m, v


def test_adamw_step_kernel_matches_ref():
    n = 1024
    p0 = cp.random.randn(n, dtype=cp.float32)
    g0 = cp.random.randn(n, dtype=cp.float32)
    m0 = cp.zeros(n, dtype=cp.float32)
    v0 = cp.zeros(n, dtype=cp.float32)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    bc1, bc2 = 1 - b1, 1 - b2

    p, m, v = p0.copy(), m0.copy(), v0.copy()
    adamw_step(p, g0, m, v, lr, b1, b2, eps, bc1, bc2, wd)

    p_ref, m_ref, v_ref = _ref_step(p0.copy(), g0, m0.copy(), v0.copy(),
                                    lr, b1, b2, eps, bc1, bc2, wd)
    assert cp.allclose(p, p_ref, rtol=1e-5, atol=1e-6)
    assert cp.allclose(m, m_ref, rtol=1e-5, atol=1e-6)
    assert cp.allclose(v, v_ref, rtol=1e-5, atol=1e-6)


def test_optimizer_dispatch_matches_fallback():
    cp.random.seed(0)
    def _new():
        return [Matrix(cp.random.randn(64, 32, dtype=cp.float32)),
                Matrix(cp.random.randn(32, dtype=cp.float32))]
    grads = [cp.random.randn(64, 32, dtype=cp.float32),
             cp.random.randn(32, dtype=cp.float32)]

    K.USE_CUSTOM_ADAMW = False
    p_off = _new()
    for p, g in zip(p_off, grads): p.grad[:] = g
    opt_off = AdamW(p_off, lr=1e-3, weight_decay=0.01)
    opt_off.step()

    cp.random.seed(0)
    K.USE_CUSTOM_ADAMW = True
    try:
        p_on = _new()
        for p, g in zip(p_on, grads): p.grad[:] = g
        opt_on = AdamW(p_on, lr=1e-3, weight_decay=0.01)
        opt_on.step()
    finally:
        K.USE_CUSTOM_ADAMW = False

    for a, b in zip(p_off, p_on):
        assert cp.allclose(a.data, b.data, rtol=1e-5, atol=1e-6)
