// dx = rstd * (dh - mean(dh) - y_hat * mean(dh * y_hat)), where dh = dy * gamma

extern "C" {

__device__ inline float block_reduce_sum_ln(float val, float* smem, int tid, int bsz) {
    smem[tid] = val;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__global__ void layernorm_fwd(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              const float* __restrict__ beta,
                              float* __restrict__ y,
                              float* __restrict__ mean_out,
                              float* __restrict__ rstd_out,
                              int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    extern __shared__ float smem[];

    const float* xr = x + (long long)row * D;
    float* yr = y + (long long)row * D;

    float s = 0.0f;
    for (int j = tid; j < D; j += bsz) s += xr[j];
    s = block_reduce_sum_ln(s, smem, tid, bsz);
    float mean = s / (float)D;

    float v = 0.0f;
    for (int j = tid; j < D; j += bsz) {
        float d = xr[j] - mean;
        v += d * d;
    }
    v = block_reduce_sum_ln(v, smem, tid, bsz);
    float var = v / (float)D;
    float rstd = rsqrtf(var + eps);

    for (int j = tid; j < D; j += bsz) {
        yr[j] = ((xr[j] - mean) * rstd) * gamma[j] + beta[j];
    }
    if (tid == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
}

__global__ void layernorm_bwd_dx(const float* __restrict__ dy,
                                 const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ mean,
                                 const float* __restrict__ rstd,
                                 float* __restrict__ dx,
                                 int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    extern __shared__ float smem[];

    const float* xr = x + (long long)row * D;
    const float* dyr = dy + (long long)row * D;
    float* dxr = dx + (long long)row * D;
    float m = mean[row];
    float r = rstd[row];

    float a = 0.0f, b = 0.0f;
    for (int j = tid; j < D; j += bsz) {
        float yh = (xr[j] - m) * r;
        float dh = dyr[j] * gamma[j];
        a += dh;
        b += dh * yh;
    }
    a = block_reduce_sum_ln(a, smem, tid, bsz);
    __syncthreads();
    b = block_reduce_sum_ln(b, smem, tid, bsz);
    float invD = 1.0f / (float)D;
    a *= invD;
    b *= invD;

    for (int j = tid; j < D; j += bsz) {
        float yh = (xr[j] - m) * r;
        float dh = dyr[j] * gamma[j];
        dxr[j] = r * (dh - a - yh * b);
    }
}

} // extern "C"
