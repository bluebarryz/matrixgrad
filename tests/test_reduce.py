import cupy as cp
import pytest
from cudagrad.kernels.bindings import row_reduce


@pytest.mark.parametrize("shape", [(64, 32), (128, 1024), (1, 4096), (33, 7)])
@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_row_reduce(shape, op):
    x = cp.random.randn(*shape, dtype=cp.float32)
    y_kernel = row_reduce(x, op).reshape(-1)
    ref = {"sum": x.sum(axis=-1), "mean": x.mean(axis=-1), "max": x.max(axis=-1)}[op]
    rtol = 1e-3 if op == "mean" and shape[1] > 1000 else 1e-5
    assert cp.allclose(y_kernel, ref, rtol=rtol, atol=1e-4)
