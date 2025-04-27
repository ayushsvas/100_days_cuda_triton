#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 2
#define BlOCK_WIDTH 2

__global__ void tiled_matmul(float *a, float *b, float *c, int width){
    __shared__ float sda[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sdb[TILE_WIDTH][TILE_WIDTH];

    // local indices in a block
    int tx = threadIdx.x; int ty =  threadIdx.y;
    int bx = blockIdx.x; int by =  blockIdx.y;

    // absolute row and col index
    int row = ty + TILE_WIDTH*by;
    int col = tx + TILE_WIDTH*bx;

    // auto variable in the register per thread
    float dvalue=0;

    // transfer values from global to shared in phases 
    for (int m=0; m< width/TILE_WIDTH; ++m){
        sda[ty][tx] = a[row*width+m*TILE_WIDTH+tx]; 
        sdb[ty][tx] = b[(m*TILE_WIDTH+ty)*width+col];
    
    __syncthreads();

        // multiply 

        for (int k=0; k<TILE_WIDTH; ++k){
            dvalue += sda[ty][k]*sdb[k][tx];
        }
    __syncthreads();

    }
    // write back row major 
    c[row*width+col] = dvalue;

}


int main(){
    const int size = 16;
    float a[size][size], b[size][size], c[size][size];
    for (int i=0; i<size; ++i){
        for (int j=0; j<size;++j){
            a[i][j] = 1;
            b[i][j] = 1;
        }
    }

    // allocate 
    float *da, *db, *dc;
    cudaMalloc(&da, size*size*sizeof(float));
    cudaMalloc(&db, size*size*sizeof(float));
    cudaMalloc(&dc, size*size*sizeof(float));


    // copy
    cudaMemcpy(da, a, size*size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size*size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 ThreadBlock(BlOCK_WIDTH, BlOCK_WIDTH);
    dim3 Grids((BlOCK_WIDTH+size-1/BlOCK_WIDTH),(BlOCK_WIDTH+size-1/BlOCK_WIDTH));

    tiled_matmul <<< Grids, ThreadBlock >>> (da, db, dc, size);

    cudaMemcpy(c, dc, size*size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    for (int i=0; i<size; ++i){
        for (int j=0; j<size; ++j){
            std::cout << c[i][j]<<" ";
        }
        std::cout <<"\n";
    }

}


