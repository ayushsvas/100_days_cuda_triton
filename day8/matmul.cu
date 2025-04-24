#include <cuda_runtime.h>
#include <iostream>
__global__  void matmul(float *a, float *b, float *c ,int Nx, int Ny){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tidx >= Nx || tidy >= Ny) return;

    float sum = 0;
    for (int k = 0; k < Ny; ++k){
        sum += a[k+Ny*tidx]*b[k*Nx+tidx];
    }

    c[tidx*Nx+tidy] = sum;
    
} 


int main(){
    const int Nx = 32;
    const int Ny = 32;

    float arr[Ny][Nx];
    float brr[Ny][Nx];
    float crr[Ny][Nx];
    for (int i=0; i< Nx; ++i){
        for (int j=0; j<Ny;++j){
            arr[i][j]=1;
            brr[i][j]=1;
        }
    }

    float *d_arr, *d_brr, *d_crr;
    cudaMalloc(&d_arr, Nx*Ny*sizeof(float));
    cudaMalloc(&d_brr, Nx*Ny*sizeof(float));
    cudaMalloc(&d_crr, Nx*Ny*sizeof(float));

    cudaMemcpy(d_arr, arr, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_brr, brr, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);

    dim3 numThreads(16,16);
    dim3 numBlocks((Nx+16-1)/16, (Ny+16-1)/16);
    // dim3 numBlocks(1,1);

    matmul <<< numBlocks, numThreads >>> (d_arr, d_brr, d_crr, Nx, Ny);

    cudaMemcpy(crr, d_crr, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<Ny; ++i){
        for (int j=0; j<Nx; ++j){
            std::cout << crr[i][j] <<" ";
        }
        std::cout <<"\n";
    }
    cudaFree(d_arr);
    cudaFree(d_brr);
    cudaFree(d_crr);



}