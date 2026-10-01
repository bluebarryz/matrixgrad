extern "C" {

// op_id: 0 = sum, 1 = mean, 2 = max
__global__ void row_reduce(const float* __restrict__ x, float* __restrict__ y,
                           int N, int D, int op_id) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    const float* row_ptr = x + (long long)row * D;

    float acc;
    if (op_id == 2) {
        acc = -INFINITY;
        for (int j = tid; j < D; j += bsz) acc = fmaxf(acc, row_ptr[j]);
    } else {
        acc = 0.0f;
        for (int j = tid; j < D; j += bsz) acc += row_ptr[j];
    }
    smem[tid] = acc;
    __syncthreads();

    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op_id == 2) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            else            smem[tid] = smem[tid] + smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float r = smem[0];
        if (op_id == 1) r /= (float)D;
        y[row] = r;
    }
}

} // extern "C"
