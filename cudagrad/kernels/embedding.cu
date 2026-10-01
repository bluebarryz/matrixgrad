extern "C" {

__global__ void embedding_gather(const float* __restrict__ table,
                                 const int* __restrict__ idx,
                                 float* __restrict__ out,
                                 int M, int D) {
    int row = blockIdx.y;
    if (row >= M) return;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;
    int t = idx[row];
    out[(long long)row * D + col] = table[(long long)t * D + col];
}

__global__ void embedding_scatter_add(float* __restrict__ dst,
                                      const int* __restrict__ idx,
                                      const float* __restrict__ src,
                                      int M, int D) {
    int row = blockIdx.y;
    if (row >= M) return;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;
    int t = idx[row];
    atomicAdd(&dst[(long long)t * D + col], src[(long long)row * D + col]);
}

} // extern "C"
