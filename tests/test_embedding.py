import cupy as cp
from cudagrad.kernels.bindings import embedding_gather, embedding_scatter_add
from cudagrad.backend import K
from cudagrad.engine import Matrix


def test_gather_2d_idx():
    V, D = 50, 16
    table = cp.random.randn(V, D, dtype=cp.float32)
    idx = cp.random.randint(0, V, size=(4, 8), dtype=cp.int32)
    out = embedding_gather(table, idx)
    assert out.shape == (4, 8, D)
    assert cp.allclose(out, table[idx])


def test_scatter_add_handles_repeats():
    V, D = 8, 4
    dst = cp.zeros((V, D), dtype=cp.float32)
    idx = cp.array([0, 0, 0, 3, 3, 7], dtype=cp.int32)
    src = cp.ones((6, D), dtype=cp.float32)
    embedding_scatter_add(dst, idx, src)
    expected = cp.zeros((V, D), dtype=cp.float32)
    expected[0] = 3; expected[3] = 2; expected[7] = 1
    assert cp.allclose(dst, expected)


def test_engine_path_parity():
    cp.random.seed(0)
    V, D = 30, 8
    table_data = cp.random.randn(V, D, dtype=cp.float32)
    idx = cp.random.randint(0, V, size=(5,), dtype=cp.int32)

    K.USE_CUSTOM_EMBEDDING = False
    a = Matrix(table_data.copy())
    ya = a[idx]; ya.grad = cp.ones_like(ya.data); ya._backward()

    K.USE_CUSTOM_EMBEDDING = True
    try:
        b = Matrix(table_data.copy())
        yb = b[idx]; yb.grad = cp.ones_like(yb.data); yb._backward()
    finally:
        K.USE_CUSTOM_EMBEDDING = False

    assert cp.allclose(ya.data, yb.data)
    assert cp.allclose(a.grad, b.grad)
