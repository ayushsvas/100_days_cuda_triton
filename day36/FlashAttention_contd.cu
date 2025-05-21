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
    
    // these will be stored in registers. Every thread with its own copy
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[HIDDEN_DIM] = {0.0f};
    
    // Below we traverse through the query block in blocks of size BLOCK_SIZExHIDDEN_DIM
    // This assigning happens in parallel (hopefully) for all the threads in BLOCK_SIZExHIDDEN_DIM
    // Only 1 tile (tile == 0) in the shared memory at a time.
    for (int tile = 0; tile < (N_inp + BLOCK_SIZE - 1)/BLOCK_SIZE; ++tile) {
        if (tile == 0 && row < N_out) {
            for (int d = 0; d < HIDDEN_DIM; d += blockDim.y) {
                int d_idx = d + ty;
                if (d_idx < HIDDEN_DIM) {
                    q_block[tx][d_idx] = query[row*HIDDEN_DIM+d_idx]; // in memory things are flattened
                }
            }
        }
        __syncthreads();
        
        if (tile * BLOCK_SIZE + ty < N_inp && row < N_out) {
            for (int d = 0; d < HIDDEN_DIM; d += blockDim.y) {
                int d_idx = d + ty;
                if (d_idx < HIDDEN_DIM) {
                    k_block[tx][d_idx] = key[(tile * BLOCK_SIZE+tx)*HIDDEN_DIM+d_idx];
                    v_block[tx][d_idx] = value[(tile * BLOCK_SIZE+tx)*HIDDEN_DIM+d_idx];
                }
            }
        }
        __syncthreads();
        