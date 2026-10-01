import os
import cupy as cp


_THIS_DIR = os.path.dirname(__file__)


def _load(name: str) -> cp.RawModule:
    with open(os.path.join(_THIS_DIR, name)) as f:
        return cp.RawModule(code=f.read(), options=("-std=c++17",), backend="nvrtc")


_elementwise = _load("elementwise.cu")
_reduce = _load("reduce.cu")
_sgemm = _load("sgemm.cu")
_softmax = _load("softmax.cu")
_layernorm = _load("layernorm.cu")
_cross_entropy = _load("cross_entropy.cu")
_embedding = _load("embedding.cu")
_adamw = _load("adamw.cu")


_UNARY_KERNELS = {
    "exp": _elementwise.get_function("unary_exp"),
    "log": _elementwise.get_function("unary_log"),
    "relu": _elementwise.get_function("unary_relu"),
}
_BINARY_KERNELS = {
    "add": _elementwise.get_function("binary_add"),
    "mul": _elementwise.get_function("binary_mul"),
}
_UNARY_POW = _elementwise.get_function("unary_pow_scalar")


def _launch_1d(n: int, threads: int = 256, max_blocks: int = 65535):
    blocks = min((n + threads - 1) // threads, max_blocks)
    return (blocks,), (threads,)


def elementwise_unary(x: cp.ndarray, op: str) -> cp.ndarray:
    assert x.dtype == cp.float32
    if not x.flags["C_CONTIGUOUS"]: x = cp.ascontiguousarray(x)
    y = cp.empty_like(x)
    n = x.size
    grid, block = _launch_1d(n)
    _UNARY_KERNELS[op](grid, block, (x, y, n))
    return y


def elementwise_pow(x: cp.ndarray, p: float) -> cp.ndarray:
    assert x.dtype == cp.float32
    if not x.flags["C_CONTIGUOUS"]: x = cp.ascontiguousarray(x)
    y = cp.empty_like(x)
    n = x.size
    grid, block = _launch_1d(n)
    _UNARY_POW(grid, block, (x, cp.float32(p), y, n))
    return y


def elementwise_binary(a: cp.ndarray, b: cp.ndarray, op: str) -> cp.ndarray:
    assert a.shape == b.shape and a.dtype == b.dtype == cp.float32
    if not a.flags["C_CONTIGUOUS"]: a = cp.ascontiguousarray(a)
    if not b.flags["C_CONTIGUOUS"]: b = cp.ascontiguousarray(b)
    y = cp.empty_like(a)
    n = a.size
    grid, block = _launch_1d(n)
    _BINARY_KERNELS[op](grid, block, (a, b, y, n))
    return y


_ROW_REDUCE = _reduce.get_function("row_reduce")
_OP_IDS = {"sum": 0, "mean": 1, "max": 2}


def row_reduce(x: cp.ndarray, op: str) -> cp.ndarray:
    assert x.ndim == 2 and x.dtype == cp.float32 and x.flags["C_CONTIGUOUS"]
    N, D = x.shape
    y = cp.empty((N, 1), dtype=cp.float32)
    threads = 256
    smem_bytes = threads * 4
    _ROW_REDUCE((N,), (threads,), (x, y, N, D, _OP_IDS[op]), shared_mem=smem_bytes)
    return y


_SGEMM_NAIVE = _sgemm.get_function("sgemm_naive")
_SGEMM_TILED = _sgemm.get_function("sgemm_tiled")

# must match sgemm.cu
_BLOCK_M = 128
_BLOCK_N = 128


def sgemm(A: cp.ndarray, B: cp.ndarray, *, kernel: str = "tiled") -> cp.ndarray:
    if A.ndim > 2 or B.ndim > 2:
        a_batch = A.reshape(-1, *A.shape[-2:]) if A.ndim > 2 else A[None]
        b_batch = B.reshape(-1, *B.shape[-2:]) if B.ndim > 2 else B[None]
        bsz = max(a_batch.shape[0], b_batch.shape[0])
        if a_batch.shape[0] == 1: a_batch = cp.broadcast_to(a_batch, (bsz, *A.shape[-2:]))
        if b_batch.shape[0] == 1: b_batch = cp.broadcast_to(b_batch, (bsz, *B.shape[-2:]))
        M, N = A.shape[-2], B.shape[-1]
        out = cp.empty((bsz, M, N), dtype=cp.float32)
        for i in range(bsz):
            out[i] = sgemm(cp.ascontiguousarray(a_batch[i]),
                           cp.ascontiguousarray(b_batch[i]), kernel=kernel)
        return out.reshape(*A.shape[:-2] if A.ndim > 2 else B.shape[:-2], M, N)

    assert A.ndim == 2 and B.ndim == 2
    assert A.dtype == cp.float32 and B.dtype == cp.float32
    if not A.flags["C_CONTIGUOUS"]: A = cp.ascontiguousarray(A)
    if not B.flags["C_CONTIGUOUS"]: B = cp.ascontiguousarray(B)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"matmul shape mismatch: ({M},{K}) @ ({K2},{N})"
    C = cp.empty((M, N), dtype=cp.float32)

    if kernel == "naive":
        block = (16, 16)
        grid = ((N + 15) // 16, (M + 15) // 16)
        _SGEMM_NAIVE(grid, block, (A, B, C, M, N, K))
    elif kernel == "tiled":
        block = (16, 16)
        grid = ((N + _BLOCK_N - 1) // _BLOCK_N, (M + _BLOCK_M - 1) // _BLOCK_M)
        _SGEMM_TILED(grid, block, (A, B, C, M, N, K))
    else:
        raise ValueError(f"unknown sgemm kernel: {kernel}")
    return C


_SOFTMAX_FWD = _softmax.get_function("softmax_row_fwd")
_SOFTMAX_BWD = _softmax.get_function("softmax_row_bwd")


def softmax_row_fwd(z: cp.ndarray) -> cp.ndarray:
    assert z.ndim == 2 and z.dtype == cp.float32 and z.flags["C_CONTIGUOUS"]
    N, D = z.shape
    y = cp.empty_like(z)
    threads = 256
    smem = threads * 4
    _SOFTMAX_FWD((N,), (threads,), (z, y, N, D), shared_mem=smem)
    return y


def softmax_row_bwd(softmax_out: cp.ndarray, dy: cp.ndarray) -> cp.ndarray:
    assert softmax_out.shape == dy.shape and softmax_out.ndim == 2
    assert softmax_out.dtype == cp.float32 == dy.dtype
    assert softmax_out.flags["C_CONTIGUOUS"] and dy.flags["C_CONTIGUOUS"]
    N, D = softmax_out.shape
    dz = cp.empty_like(softmax_out)
    threads = 256
    smem = threads * 4
    _SOFTMAX_BWD((N,), (threads,), (softmax_out, dy, dz, N, D), shared_mem=smem)
    return dz


_LN_FWD = _layernorm.get_function("layernorm_fwd")
_LN_BWD_DX = _layernorm.get_function("layernorm_bwd_dx")


def layernorm_fwd(x: cp.ndarray, gamma: cp.ndarray, beta: cp.ndarray, eps: float):
    assert x.ndim == 2 and x.dtype == cp.float32 and x.flags["C_CONTIGUOUS"]
    assert gamma.shape == beta.shape == (x.shape[1],)
    N, D = x.shape
    y = cp.empty_like(x)
    mean = cp.empty((N,), dtype=cp.float32)
    rstd = cp.empty((N,), dtype=cp.float32)
    threads = 256
    smem = threads * 4
    _LN_FWD((N,), (threads,),
            (x, cp.ascontiguousarray(gamma), cp.ascontiguousarray(beta),
             y, mean, rstd, N, D, cp.float32(eps)),
            shared_mem=smem)
    return y, mean, rstd


def layernorm_bwd(dy: cp.ndarray, x: cp.ndarray, gamma: cp.ndarray,
                  mean: cp.ndarray, rstd: cp.ndarray):
    assert dy.shape == x.shape and dy.ndim == 2
    N, D = x.shape
    if not dy.flags["C_CONTIGUOUS"]: dy = cp.ascontiguousarray(dy)
    if not x.flags["C_CONTIGUOUS"]: x = cp.ascontiguousarray(x)
    dx = cp.empty_like(x)
    threads = 256
    smem = threads * 4
    _LN_BWD_DX((N,), (threads,),
               (dy, x, cp.ascontiguousarray(gamma), mean, rstd, dx, N, D),
               shared_mem=smem)
    y_hat = (x - mean[:, None]) * rstd[:, None]
    dgamma = (dy * y_hat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    return dx, dgamma, dbeta


_CE_FUSED = _cross_entropy.get_function("cross_entropy_fused")


def cross_entropy_fused(z: cp.ndarray, targets: cp.ndarray):
    assert z.ndim == 2 and z.dtype == cp.float32 and z.flags["C_CONTIGUOUS"]
    if targets.dtype != cp.int32: targets = targets.astype(cp.int32)
    if not targets.flags["C_CONTIGUOUS"]: targets = cp.ascontiguousarray(targets)
    N, C = z.shape
    per_row = cp.empty((N,), dtype=cp.float32)
    grad_z = cp.empty_like(z)
    threads = 256
    smem = threads * 4
    _CE_FUSED((N,), (threads,), (z, targets, per_row, grad_z, N, C), shared_mem=smem)
    loss = per_row.mean()
    return loss, grad_z


_EMB_GATHER = _embedding.get_function("embedding_gather")
_EMB_SCATTER = _embedding.get_function("embedding_scatter_add")


def embedding_gather(table: cp.ndarray, idx: cp.ndarray) -> cp.ndarray:
    assert table.ndim == 2 and table.dtype == cp.float32 and table.flags["C_CONTIGUOUS"]
    out_shape = idx.shape + (table.shape[1],)
    idx_flat = idx.reshape(-1)
    if idx_flat.dtype != cp.int32: idx_flat = idx_flat.astype(cp.int32)
    if not idx_flat.flags["C_CONTIGUOUS"]: idx_flat = cp.ascontiguousarray(idx_flat)
    M = idx_flat.size
    D = table.shape[1]
    out = cp.empty((M, D), dtype=cp.float32)
    threads = 128
    grid_x = (D + threads - 1) // threads
    _EMB_GATHER((grid_x, M), (threads,), (table, idx_flat, out, M, D))
    return out.reshape(out_shape)


def embedding_scatter_add(dst: cp.ndarray, idx: cp.ndarray, src: cp.ndarray):
    assert dst.ndim == 2 and dst.dtype == cp.float32
    D = dst.shape[1]
    idx_flat = idx.reshape(-1)
    if idx_flat.dtype != cp.int32: idx_flat = idx_flat.astype(cp.int32)
    if not idx_flat.flags["C_CONTIGUOUS"]: idx_flat = cp.ascontiguousarray(idx_flat)
    src_flat = src.reshape(-1, D)
    if not src_flat.flags["C_CONTIGUOUS"]: src_flat = cp.ascontiguousarray(src_flat)
    if not dst.flags["C_CONTIGUOUS"]:
        raise ValueError("scatter_add target must be contiguous")
    M = idx_flat.size
    threads = 128
    grid_x = (D + threads - 1) // threads
    _EMB_SCATTER((grid_x, M), (threads,), (dst, idx_flat, src_flat, M, D))


_ADAMW_STEP = _adamw.get_function("adamw_step")


def adamw_step(p: cp.ndarray, g: cp.ndarray, m: cp.ndarray, v: cp.ndarray,
               lr: float, b1: float, b2: float, eps: float,
               bc1: float, bc2: float, wd: float):
    assert p.shape == g.shape == m.shape == v.shape
    assert p.dtype == cp.float32 and g.dtype == cp.float32
    for arr in (p, g, m, v):
        assert arr.flags["C_CONTIGUOUS"]
    n = p.size
    threads = 256
    blocks = min((n + threads - 1) // threads, 65535)
    _ADAMW_STEP((blocks,), (threads,),
                (p, g, m, v,
                 cp.float32(lr), cp.float32(b1), cp.float32(b2), cp.float32(eps),
                 cp.float32(bc1), cp.float32(bc2), cp.float32(wd),
                 n))
