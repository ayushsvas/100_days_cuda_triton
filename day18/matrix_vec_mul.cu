#include <cuda_runtime.h>
#include <iostream>
// Each thread is perfomring multiple operations -- mutliplying and summing across respective row with the vector V
__global__ void matrix_vec_multiply(float *A, float *V, float *C, int dims){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= dims) return ;

    float value = 0;
    for (int j = 0; j<dims; ++j){
    value += A[tidx*dims+j]*V[j];
    }
    C[tidx] = value;
}


int main(){
    int dims = 8;
    float h_A[dims][dims];
    float h_V[dims];
    float h_C[dims];

    // initialize
    for (int i = 0; i<dims; ++i){
        for (int j=0; j<dims; ++j){
            h_A[i][j] = 1;
        }
        h_V[i] = 1;
    }

    // device copies and stores
    float* d_A;
    float* d_V;
    float* d_C;

    // allocate memory
    cudaMalloc(&d_A, dims*dims*sizeof(float));
    cudaMalloc(&d_V, dims*sizeof(float));
    cudaMalloc(&d_C, dims*sizeof(float));
    
    // copy data to device
    cudaMemcpy(d_A, h_A, dims*dims*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, dims*sizeof(float), cudaMemcpyHostToDevice);

    // threads, blocks and grids 
    int numThreads = 128;
    int numBlocks = 1;

    // kernel launch
    matrix_vec_multiply <<< numBlocks, numThreads >>> (d_A, d_V, d_C, dims);

    // copy result to output vector on host 
    cudaMemcpy(h_C, d_C, dims*sizeof(float), cudaMemcpyDeviceToHost);

    // print the result
    for (int i=0; i<dims; ++i){
        std:: cout << h_C[i] <<"\n";
    }
    
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_C);

    return 0;
}

// if one wants to define a function that is visible to both host and device, one can use
// __device__ __host__ void fn(){}

// keep ThreadBlock size as multiple of 32.



