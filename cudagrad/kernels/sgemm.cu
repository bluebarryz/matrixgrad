// 128x128 tile per block, 8x8 per thread

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 8
#define TM 8
#define TN 8

extern "C" {

__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int tid = thread_row * 16 + thread_col;

    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int a_row = load_idx / BLOCK_K;
            int a_col = load_idx % BLOCK_K;
            int g_row = block_row + a_row;
            int g_col = k_tile + a_col;
            As[a_row][a_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int b_row = load_idx / BLOCK_N;
            int b_col = load_idx % BLOCK_N;
            int g_row = k_tile + b_row;
            int g_col = block_col + b_col;
            Bs[b_row][b_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[thread_row * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][thread_col * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    c_reg[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + thread_row * TM + i;
        if (g_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + thread_col * TN + j;
            if (g_col < N) C[g_row * N + g_col] = c_reg[i][j];
        }
    }
}

} // extern "C"
