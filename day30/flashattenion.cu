#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;
constexpr int HIDDEN_DIM = 128;

extern "C" __global__ void FlashAttention(
    float* output,
    float* output_lse,
    const float* query,
    const float* key,
    const float* value,
    const float scale,
    const int N_out,
    const int N_inp
) {
    __shared__ float q_block[BLOCK_SIZE][HIDDEN_DIM];
    __shared__ float k_block[BLOCK_SIZE][HIDDEN_DIM];
    __shared__ float v_block[BLOCK_SIZE][HIDDEN_DIM];
    
    const int tx = threadIdx.x; // these are local to a block
    const int ty = threadIdx.y;
    const int row = blockIdx.x * BLOCK_SIZE + tx; // getting global index
    
   
}