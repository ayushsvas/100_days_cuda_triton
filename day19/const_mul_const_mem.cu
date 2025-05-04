#include <cuda_runtime.h>
#include <iostream>

#define N 10

// Declare constant memory on the device
__constant__ float constMultiplier[N];

// Kernel that uses the constant memory
__global__ void multiplyWithConstant(const float* input, float* output) {
    int idx = threadIdx.x;
    output[idx] = input[idx] * constMultiplier[idx];
}

int main() {
    float h_input[N], h_output[N], h_multiplier[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_input[i] = float(i);           // input: 0, 1, 2, ..., 9
        h_multiplier[i] = float(i + 1);  // multiplier: 1, 2, 3, ..., 10
    }

    // Copy multiplier to constant memory
    cudaMemcpyToSymbol(constMultiplier, h_multiplier, N * sizeof(float));

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    multiplyWithConstant<<<1, N>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
