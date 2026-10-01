extern "C" {

__global__ void unary_exp(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = expf(x[i]);
}

__global__ void unary_log(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = logf(x[i]);
}

__global__ void unary_relu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void unary_pow_scalar(const float* __restrict__ x, float p, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = powf(x[i], p);
}

__global__ void binary_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = a[i] + b[i];
}

__global__ void binary_mul(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = a[i] * b[i];
}

} // extern "C"
