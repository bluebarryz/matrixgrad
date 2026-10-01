extern "C" {

__global__ void adamw_step(float* __restrict__ p,
                           const float* __restrict__ g,
                           float* __restrict__ m,
                           float* __restrict__ v,
                           float lr, float b1, float b2, float eps,
                           float bc1, float bc2, float wd,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float pi = p[i];
        if (wd != 0.0f) pi -= lr * wd * pi;
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        pi -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
        p[i] = pi;
    }
}

} // extern "C"
