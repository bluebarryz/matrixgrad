extern "C" {

__device__ inline float block_reduce_sum(float val, float* smem, int tid, int bsz) {
    smem[tid] = val;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__device__ inline float block_reduce_max(float val, float* smem, int tid, int bsz) {
    smem[tid] = val;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__global__ void softmax_row_fwd(const float* __restrict__ z, float* __restrict__ y,
                                int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    extern __shared__ float smem[];

    const float* zr = z + (long long)row * D;
    float* yr = y + (long long)row * D;

    float m = -INFINITY;
    for (int j = tid; j < D; j += bsz) m = fmaxf(m, zr[j]);
    m = block_reduce_max(m, smem, tid, bsz);

    float s = 0.0f;
    for (int j = tid; j < D; j += bsz) {
        float e = expf(zr[j] - m);
        yr[j] = e;
        s += e;
    }
    s = block_reduce_sum(s, smem, tid, bsz);

    float inv_s = 1.0f / s;
    for (int j = tid; j < D; j += bsz) yr[j] *= inv_s;
}

__global__ void softmax_row_bwd(const float* __restrict__ softmax_out,
                                const float* __restrict__ dy,
                                float* __restrict__ dz,
                                int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    extern __shared__ float smem[];

    const float* s_row = softmax_out + (long long)row * D;
    const float* dy_row = dy + (long long)row * D;
    float* dz_row = dz + (long long)row * D;

    float acc = 0.0f;
    for (int j = tid; j < D; j += bsz) acc += dy_row[j] * s_row[j];
    float dot = block_reduce_sum(acc, smem, tid, bsz);

    for (int j = tid; j < D; j += bsz) dz_row[j] = s_row[j] * (dy_row[j] - dot);
}

} // extern "C"
