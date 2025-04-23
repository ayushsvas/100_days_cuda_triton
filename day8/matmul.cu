#include <cuda_runtime.h>

__global__  void matmul(float *a, float *b, float *c ,int Nx, int Ny){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tidx >= Nx || tidy >= Ny) return;


    int outidx = tidx + tidy*Nx;

    float sum = 0;
    for (int k = 0; k<Nx; ++k){
        sum += a[k+Nx*idy]*b[idy+];
    }


    c[outidx] += a[tidx+Nx*tidy]*b[tidy+Ny*tidx];
    
    
} 