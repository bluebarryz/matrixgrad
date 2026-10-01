import cupy as cp
import pytest
from cudagrad.kernels.bindings import sgemm


@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 256, 64), (513, 513, 513), (1024, 1024, 1024)])
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_sgemm(M, N, K, kernel):
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(K, N, dtype=cp.float32)
    C_kernel = sgemm(A, B, kernel=kernel)
    C_ref = A @ B
    assert cp.allclose(C_kernel, C_ref, rtol=1e-3, atol=1e-3)


def test_sgemm_via_backend():
    from cudagrad.backend import K
    from cudagrad.engine import Matrix
    A = Matrix(cp.random.randn(128, 64, dtype=cp.float32))
    B = Matrix(cp.random.randn(64, 32, dtype=cp.float32))

    K.USE_CUSTOM_MATMUL = False
    C_ref = (A @ B).data
    K.USE_CUSTOM_MATMUL = True
    try:
        C_kernel = (A @ B).data
    finally:
        K.USE_CUSTOM_MATMUL = False
    assert cp.allclose(C_kernel, C_ref, rtol=1e-3, atol=1e-3)
