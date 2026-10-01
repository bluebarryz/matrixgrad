import cupy as cp
import time
from cudagrad.kernels.bindings import softmax_row_fwd

SHAPES = [(1024, 1024), (2048, 4096), (4096, 8192)]
WARMUP = 5
ITERS = 50


def _time(fn):
    for _ in range(WARMUP): fn(); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / ITERS


def _cupy_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = cp.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _gb_per_s(z, sec):
    return 2 * z.nbytes / sec / 1e9


print(f"{'N':>6} {'D':>6}  {'fused':>14}  {'cupy':>14}  {'speedup':>8}")
for N, D in SHAPES:
    z = cp.random.randn(N, D, dtype=cp.float32)
    t_fused = _time(lambda: softmax_row_fwd(z))
    t_cupy  = _time(lambda: _cupy_softmax(z))
    print(f"{N:>6} {D:>6}  {_gb_per_s(z, t_fused):>10.1f} GB/s  "
          f"{_gb_per_s(z, t_cupy):>10.1f} GB/s  {t_cupy/t_fused:>6.2f}x")
