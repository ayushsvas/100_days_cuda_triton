
#include <cuda_runtime.h>


__global__ void reduceSum(float *a, int *N)
__shared__ float partialSum[]
unsigned int t = threadIdx.x; 
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) 
{ 
 __syncthreads(); 
  if (t % (2*stride) == 0) {
    partialSum[t] += partialSum[t+stride]; 
  }
}