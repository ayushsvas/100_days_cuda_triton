#include <cuda_runtime.h>


#define TILE_SIZE 

__device__ void access(float *a){

}


__global__ void access_tile_stack(float *chunk){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    int gidx = idz * gridDim.z + idy * blockDim.x + idx;




 

}