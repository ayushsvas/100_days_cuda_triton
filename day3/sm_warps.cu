#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>


__global__ void sm_warps(){
    int tid = threadIdx.x;
    int blockid = blockIdx.x;
    int smid, warpid, laneid;
    

    // Below is inline PTX (Paralell Thread Exexcution) assembly (asm). 
    asm("mov.u32 %0, %smid;":"=r"(smid)); //%0 is the figurehead for general purpose register ("=r"). Move SMID (also a register) to r register. 
    asm volatile ("mov.u32 %0, %warpid;":"=r"(warpid)); //volatile is a keyword telling the optimizer not to optimize (like cache warpid value),
    asm volatile ("mov.u32 %0, %laneid;":"=r"(laneid)); //assuming it won't change.  Makes sure the value is always read and the line is executed.

    printf("SMID:%d | BlockID:%d | WarpID:%d | LaneID:%d | Thread:%d --- Namaste!\n", smid, blockid, warpid, laneid, tid);

}

/* A100 has 128 Streaming Multiprocessors (SMs). Each SM runs 1 Block of threads. 1 Block can max take 1024 threads.
   Threads in a block are batched in warps (holding 32 threads). All the threads in the warp are executed simultaneously.
   Each SM has 4 (can vary across GPU models) warp schedulers. Warp scheduling is done in cyclic non-linear fashion; not one after the other. 
*/

int main(){
    std:: cout << "Launching kernel..." << std::endl;
    dim3 NumThreadsPerBlock(2,2); // two warps 
    dim3 numBlocks(1,2); //two SMs (numBlocks = Num SMs)
    sm_warps <<< numBlocks, NumThreadsPerBlock >>> ();
    cudaDeviceSynchronize();


}



