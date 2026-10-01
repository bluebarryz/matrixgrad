import cupy as cp
import time
from cudagrad.kernels.bindings import sgemm

SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
WARMUP = 3
ITERS = 10


def _time(fn, *args):
    for _ in range(WARMUP): fn(*args); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn(*args)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / ITERS


def _tflops(M, N, K, sec):
    return 2.0 * M * N * K / sec / 1e12


print(f"{'M':>6} {'N':>6} {'K':>6}  {'naive':>10}  {'tiled':>10}  {'cuBLAS':>10}  {'% cuBLAS':>9}")
for M, N, K in SHAPES:
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(K, N, dtype=cp.float32)
    t_naive = _time(lambda: sgemm(A, B, kernel="naive"))
    t_tiled = _time(lambda: sgemm(A, B, kernel="tiled"))
    t_cubl  = _time(lambda: A @ B)
    pct = _tflops(M, N, K, t_tiled) / _tflops(M, N, K, t_cubl) * 100
    print(f"{M:>6} {N:>6} {K:>6}  {_tflops(M,N,K,t_naive):>8.2f} TF "
          f"{_tflops(M,N,K,t_tiled):>8.2f} TF "
          f"{_tflops(M,N,K,t_cubl):>8.2f} TF  {pct:>7.1f}%")
