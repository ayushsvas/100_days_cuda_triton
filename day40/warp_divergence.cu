//No cuda code today


// Started reading chapter 6 from PMPP. 
//Sec 6.1 Warp and Threads -- It talked about warp divergence due to if-else conditionals etc. leading
// to multiple calls to control unit which is SIMD (single instruction multiple data) which can make the kernel slower.
// There are multiple processing units each of which implements a thread in a warp. 
// There are some things I didn;t understand related to the extended hardware for thread divergence situations.
