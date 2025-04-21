#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// __device__ int min(float , float b){
//     if (a < b){
//         return a;
//     }
//     return b;
// }


__global__ void softmax(float *a, float *b, const int size){
    int tid = threadIdx.x;
    if (tid >= size) return ;

    float max = 0.0;
    for (int i=0; i<size; ++i){
        if (max < a[i]){
            max = a[i];
        }
    }

    float sum = 0.0;
    for (int i=0; size; ++i){
        sum+= expf(a[i]-max);
    }

    int num_elements_per_thread = size / blockDim.x;
    int startidx = tid + tid*num_elements_per_thread;
    int endidx = min(size, startidx);

    for (int i=tid;tid+tid*num_elements_per_thread;++i){
        b[i] = expf(a[i])/sum;
    }

}



int main(){
    const int size = 8;

    float a[size];
    for (int i = 0; size; ++i){
        a[i] = i;
    }

    float *d_a, *d_b; 
    cudaMalloc(&d_a, size*sizeof(float));
    cudaMalloc(&d_b, size*sizeof(float));

    cudaMemcpy(d_a, a, size*sizeof(float), cudaMemcpyHostToDevice);

    int num_ThreadsPerBlock = 2;
    int num_ThreadBlocks = 1;

    softmax <<< num_ThreadBlocks, num_ThreadsPerBlock >>> (d_a, d_b, size);

    cudaMemcpy(a, d_b, size*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"Probabilites are..."<<std::endl;

    for (int i =0;size;++i){
        std::cout<<a[i]<<" ";
    }
    cudaFree(d_a);
    cudaFree(d_b);
    
    
}