#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);  // Get the current device

    // Query maximum number of threads per block
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
    printf("Max threads per block: %d\n", maxThreadsPerBlock);

    // Query maximum number of blocks per grid (in each dimension)
    int maxGridDimX, maxGridDimY, maxGridDimZ;
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, device);
    cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, device);
    cudaDeviceGetAttribute(&maxGridDimZ, cudaDevAttrMaxGridDimZ, device);
    printf("Max grid dimensions: X = %d, Y = %d, Z = %d\n", maxGridDimX, maxGridDimY, maxGridDimZ);

    // Query maximum block dimensions (in each dimension)
    int maxBlockDimX, maxBlockDimY, maxBlockDimZ;
    cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);
    cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device);
    cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, device);
    printf("Max block dimensions: X = %d, Y = %d, Z = %d\n", maxBlockDimX, maxBlockDimY, maxBlockDimZ);


    // Query the maximum shared memory per block
    int maxSharedMemoryPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
    printf("Max shared memory per block: %d KB\n", maxSharedMemoryPerBlock/(1024));

    // Query the maximum shared memory per multiprocessor (SM)
    int maxSharedMemoryPerMultiprocessor;
    cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    printf("Max shared memory per multiprocessor: %d KB\n", maxSharedMemoryPerMultiprocessor/(1024));

    // Query the maximum number of registers per block
    int maxRegistersPerBlock;
    cudaDeviceGetAttribute(&maxRegistersPerBlock, cudaDevAttrMaxRegistersPerBlock, device);
    printf("Max registers per block: %d\n", maxRegistersPerBlock);

    // Query the maximum number of registers per multiprocessor
    int maxRegistersPerMultiprocessor;
    cudaDeviceGetAttribute(&maxRegistersPerMultiprocessor, cudaDevAttrMaxRegistersPerMultiprocessor, device);
    printf("Max registers per multiprocessor: %d\n", maxRegistersPerMultiprocessor);


    return 0;
}
