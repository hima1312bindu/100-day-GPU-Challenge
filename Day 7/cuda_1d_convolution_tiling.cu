#include <iostream>
#include <cuda_runtime.h>

#define Mask_width 5  
__constant__ float M[Mask_width];

__global__ void oned_convolution_tiling_kernel(const float* A, float* C, int n) {
    int threadId = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadId;

    __shared__ float S_A[32 + Mask_width - 1];

    if (i < n) {
        S_A[threadId + Mask_width/2] = A[i];
    }

    if (threadId < Mask_width/2) {
        int left_idx = blockIdx.x * blockDim.x - (Mask_width/2) + threadId;
        S_A[threadId] = (left_idx >= 0) ? A[left_idx] : 0.0f;
    }

    if (threadId < Mask_width/2) {
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadId;
        S_A[threadId + blockDim.x + Mask_width/2] = (right_idx < n) ? A[right_idx] : 0.0f;
    }

    __syncthreads();

    if (i < n) {
        float result = 0.0f;
        for (int k = 0; k < Mask_width; k++) {
            int idx = threadId + k;
            int global_idx = i + k - Mask_width/2;
            if (global_idx >= 0 && global_idx < n) {
                result += S_A[idx] * M[k];
            }
        }
        C[i] = result;
    }
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 10;
    float A[n], C[n];
    float d_M[Mask_width];

    for (int i = 0; i < Mask_width; i++) d_M[i] = i;
    for (int i = 0; i < n; i++) A[i] = i;

    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, A, n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");
    cudaMemcpyToSymbol(M, d_M, Mask_width * sizeof(float));
    checkCudaError("Failed to copy mask data to device");

    dim3 dimBlock(32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
    oned_convolution_tiling_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n);
    checkCudaError("Failed to execute the kernel");

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    cudaFree(d_a);
    cudaFree(d_c);

    printf("A:\n");
    for (int i = 0; i < n; i++) printf("%.2f ", A[i]);
    printf("\n");

    printf("Mask:\n");
    for (int i = 0; i < Mask_width; i++) printf("%.2f ", d_M[i]);
    printf("\n");

    printf("C:\n");
    for (int i = 0; i < n; i++) printf("%.2f ", C[i]);
    printf("\n");

    return 0;
}
