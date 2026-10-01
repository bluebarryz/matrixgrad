import cupy as cp
import pytest
from cudagrad.kernels.bindings import (
    elementwise_unary, elementwise_pow, elementwise_binary,
)


@pytest.mark.parametrize("shape", [(1024,), (128, 256), (32, 32, 32)])
@pytest.mark.parametrize("op,ref", [
    ("exp", cp.exp),
    ("log", cp.log),
    ("relu", lambda x: cp.maximum(0, x)),
])
def test_unary(shape, op, ref):
    x = cp.random.randn(*shape, dtype=cp.float32)
    if op == "log":
        x = cp.abs(x) + 0.1
    y_kernel = elementwise_unary(x, op)
    assert cp.allclose(y_kernel, ref(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [-1.0, 0.5, 2.0, 3.0])
def test_pow(p):
    x = cp.abs(cp.random.randn(128, 64, dtype=cp.float32)) + 0.1
    assert cp.allclose(elementwise_pow(x, p), cp.power(x, p), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op,ref", [("add", lambda a, b: a + b), ("mul", lambda a, b: a * b)])
def test_binary(op, ref):
    a = cp.random.randn(256, 128, dtype=cp.float32)
    b = cp.random.randn(256, 128, dtype=cp.float32)
    assert cp.allclose(elementwise_binary(a, b, op), ref(a, b), rtol=1e-5, atol=1e-5)
