#include <cuda_runtime.h>
#include <iostream>


#define TILE_SIZE 2
#define BLOCK_SIZE 2

__global__ void tiled_transpose(float *a, int height, int width){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx >= width || ty >= height) return;

    int idx = ty*width + tx;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    tile[threadIdx.x][threadIdx.y] = a[idx];
    a[idx] = tile[threadIdx.y][threadIdx.x];

}

int main(){
    int N = 32;
    float h_a[N][N];
    for (int i=0; i<N; ++i){
        for (int j=0; j<N; ++j){
            if (i % 2 == 0){
                h_a[i][j] = 0;
            }
            else {
                h_a[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < N; ++i){
        for (int j = 0; j<N; ++j){
            std::cout << h_a[i][j] << " ";
        }
        std::cout << "\n";
    }


    float *d_a;
    cudaMalloc(&d_a, N*N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 NumThreadBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 NumGrids((N+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);

    tiled_transpose <<< NumGrids, NumThreadBlocks >>> (d_a, N, N);

    cudaMemcpy(h_a, d_a, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i){
        for (int j = 0; j<N; ++j){
            std::cout << h_a[i][j] << " ";
        }
        std::cout << "\n";
    }


}