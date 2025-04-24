// #include <torch/torch.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

__global__ void matadd_kernel(const float* a, const float* b, float* c, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int Nx = 32;
    const int Ny = 32;

    // Allocate tensors on GPU
    torch::Tensor a = torch::ones({Ny, Nx}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor b = torch::ones({Ny, Nx}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor c = torch::zeros({Ny, Nx}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Launch CUDA kernel
    dim3 threads(16, 16);
    dim3 blocks((Nx + 15) / 16, (Ny + 15) / 16);

    matadd_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        Nx, Ny
    );

    // Sync and copy result to CPU
    cudaDeviceSynchronize();
    torch::Tensor c_cpu = c.cpu();

    std::cout << "Result matrix:\n";
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            std::cout << c_cpu[i][j].item<float>() << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
