#include <iostream>
#include <cmath>

// nvcc cuda.cu -Xcompiler=-fPIC -g -gencode arch=compute_12,code=sm_12

__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] += x[i];
    }
}

__global__
void init(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        x[i] = i;
        y[i] = i;
    }
}

#define CHECK(statement) { \
    cudaError_t res = (statement); \
    if (res != 0) { \
        std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << ": "<< #statement << ": " << cudaGetErrorString(res) << "\n"; \
        exit(1); \
    } \
}

int main(void) {
    int N = 1<<20;

    float *x;
    CHECK(cudaMalloc(&x, N * sizeof(float)));

    float *y;
    CHECK(cudaMalloc(&y, N * sizeof(float)));

    struct { float *x, *y; } host;
    CHECK(cudaMallocHost(&host.x, sizeof(float) * N));
    CHECK(cudaMallocHost(&host.y, sizeof(float) * N));

    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < 1<<20; ++i) {
        init<<<numBlocks, blockSize>>>(N, x, y);
        CHECK(cudaDeviceSynchronize());

        add<<<numBlocks, blockSize>>>(N, x, y);
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaMemcpy(host.x, x, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(host.y, y, sizeof(float) * N, cudaMemcpyDeviceToHost));

    cudaFree(x);
    cudaFree(y);
    cudaFreeHost(host.x);
    cudaFreeHost(host.y);

    return 0;
}