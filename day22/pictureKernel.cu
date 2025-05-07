/*
1D indexing - 
tid = blockIdx.x*blockDim.x + threadIdx.x;

2D indexing - 
tidx = blockIdx.x*blockDim.x + threadIdx.x;
tidy = blockIdx.y*blockDim.y + threadIdx.y;

3D indexing 
tidx = threadIdx.x + blockIdx.x * blockDim.x;
tidy = threadIdx.y + blockIdx.y * blockDim.y;
tidz or (call it (select) plane) = threadIdx.z + blockIdx.z * blockDim.z;


__syncthreads() only works for all threads in a block. All threads within a block share the same execution (SM) resources.
There is no sync between threads between different blocks. This basically allows to scale the program with hardware as more the number of blocks allowed 
to run in parallel (implying allowed async between thread blocks), lesser the program takes to finish. (transparent scalabilty)

An SM can handle multiple blocks (8 max according to PMPP). According to PMPP, 1536 threads can be assinged to an SM. So can't do 12 blocks of 128 threads each 
since 12 > 8. 






*/