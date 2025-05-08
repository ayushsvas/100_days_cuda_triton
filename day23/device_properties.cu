#include <cuda_runtime.h>


// Quering device properties -- number of devices, max threads per block etc. etc.
// Schduling threads and warps -- Usually an SM can handle only certain (8 blocks) number of blocks
// and certain number of threads (1536), I guess in compute capabolity 3.0.
// Warps in a block are 32 threads with IDs 0-31, 32-63 (if there are 64 threads in a block).
// A latency hiding mechanism is set in place where 'free' warps are scheduled before warps which are
// requiring to wait for some long latency operation like floating-point arithmetics, conditionals, global 
// memory accesses