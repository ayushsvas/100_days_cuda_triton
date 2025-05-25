#include <cuda_runtime.h>


#define TILE_SIZE_XY 512
#define TILE_SIZE_Z 5


__device__ void access(float *a){

}


__global__ void access_tile_stack(float *chunk, int len_z, int len_x, int len_y){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    int gidx = idz * (len_x * len_y) + idy * len_x + idx;

 
}