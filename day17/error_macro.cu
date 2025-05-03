#include <cuda_runtime.h>

# define ERROR_CHECK(call){\
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
    printf("%s in %s at line %d \n", \
        cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
    }\
} 


// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in line %d: %s\n", __LINE__, \
                cudaGetErrorString(err)); \
        return err; \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult_t err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT error in line %d: %d\n", __LINE__, err); \
        return cudaErrorUnknown; \
    } \
}

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error\n";                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
