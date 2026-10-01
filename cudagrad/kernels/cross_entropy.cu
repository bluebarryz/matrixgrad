// grad_z = (softmax(z) - onehot(target)) / N

extern "C" {

__device__ inline float block_reduce_sum_ce(float val, float* smem, int tid, int bsz) {
    smem[tid] = val;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__device__ inline float block_reduce_max_ce(float val, float* smem, int tid, int bsz) {
    smem[tid] = val;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__global__ void cross_entropy_fused(const float* __restrict__ z,
                                    const int* __restrict__ targets,
                                    float* __restrict__ per_row_loss,
                                    float* __restrict__ grad_z,
                                    int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    extern __shared__ float smem[];

    const float* zr = z + (long long)row * C;
    float* gr = grad_z + (long long)row * C;
    int t = targets[row];

    float m = -INFINITY;
    for (int j = tid; j < C; j += bsz) m = fmaxf(m, zr[j]);
    m = block_reduce_max_ce(m, smem, tid, bsz);
    __syncthreads();

    float s = 0.0f;
    for (int j = tid; j < C; j += bsz) {
        float e = expf(zr[j] - m);
        gr[j] = e;
        s += e;
    }
    s = block_reduce_sum_ce(s, smem, tid, bsz);

    if (tid == 0) {
        float log_sum_exp = m + logf(s);
        per_row_loss[row] = log_sum_exp - zr[t];
    }

    float inv_s = 1.0f / s;
    float inv_N = 1.0f / (float)N;
    for (int j = tid; j < C; j += bsz) gr[j] = (gr[j] * inv_s) * inv_N;
    __syncthreads();
    if (tid == 0) gr[t] -= inv_N;
}

} // extern "C"
