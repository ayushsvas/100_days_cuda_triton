// #include <cuda_runtime.h>
// #include <cuComplex.h> 
// #include <math.h>
// #include <torch/extension.h>
// #include <cufft.h>

// #define EPSILON 2.2204e-16
// #define PI 3.14159265358979323846

// // Kernel for setting aperture (making particles opaque)
// __global__ void set_aperture_kernel(cuDoubleComplex* field, 
//                                    const double* x, const double* y, const double* r,
//                                    int particle_idx, int N_x, int N_y) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     if (idx >= N_x || idy >= N_y) {
//         return;
//     }
    
//     // Center coordinates at N/2
//     double x_cen = (double)(idx - N_x/2);
//     double y_cen = (double)(idy - N_y/2);
    
//     // Check if point is inside particle
//     double rho = (x_cen - x[particle_idx]) * (x_cen - x[particle_idx]) + 
//                  (y_cen - y[particle_idx]) * (y_cen - y[particle_idx]);
                 
//     if (rho < r[particle_idx] * r[particle_idx]) {
//         // If point is inside particle, set to zero (opaque)
//         field[idx * N_y + idy] = make_cuDoubleComplex(0.0, 0.0);
//     }
// }

// // Kernel for applying propagation transfer function
// __global__ void apply_propagation_kernel(cuDoubleComplex* field, 
//                                         const double* kernel, 
//                                         double propagation_distance,
//                                         int N_x, int N_y) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     if (idx >= N_x || idy >= N_y) {
//         return;
//     }
    
//     int index = idx * N_y + idy;
        
//     double kernel_val = kernel[index] * propagation_distance;
//     cuDoubleComplex modified_kernel = make_cuDoubleComplex(cos(kernel_val), sin(kernel_val));
 
//     // Multiply field with kernel using complex multiplication
//     cuDoubleComplex field_val = field[index];
//     cuDoubleComplex result;
    
//     // result = cuCmul(field_val, modified_kernel);
//     result.x = field_val.x * modified_kernel.x - field_val.y * modified_kernel.y;
//     result.y = field_val.x * modified_kernel.y + field_val.y * modified_kernel.x;
    
//     field[index] = result;
// }

// // Kernel for normalizing FFT results
// __global__ void normalize_fft_kernel(cuDoubleComplex* field, int N_x, int N_y, bool forward) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     if (idx >= N_x || idy >= N_y) {
//         return;
//     }
    
//     int index = idx * N_y + idy;
    
//     if (forward) {
//         // No normalization needed for forward FFT in cuFFT
//         return;
//     } else {
//         // Normalize inverse FFT by 1/(N_x*N_y)
//         double scale = 1.0 / (N_x * N_y);
//         field[index].x *= scale;
//         field[index].y *= scale;
//     }
// }

// // Generate the angular spectrum propagation kernel on the device
// __global__ void generate_asm_kernel(double* kernel, 
//                                    const double* fx, const double* fy,
//                                    double lambda, int N_x, int N_y) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
//     if (idx >= N_x || idy >= N_y) {
//         return;
//     }
    
//     int index = idx * N_y + idy;
    
//     // Calculate propagation term
//     double term = 1.0 - lambda * lambda * (fx[idx] * fx[idx] + fy[idy] * fy[idy]);
//     double arg = 2.0 * PI * sqrt(term) / lambda;
//     kernel[index] = arg;

//     // Check for evanescent waves
//     // if (term < 0) {
//     //     kernel[index] = 0; // Evanescent waves
//     // } else {
//         // kernel[index] = arg;
//     // }
// }

// // Main function for angular spectrum method
// cudaError_t asm_propagate(torch::Tensor& x, torch::Tensor& y, torch::Tensor& z,
//     torch::Tensor& r, int N_x, int N_y, double lambda, double dx, double dy,
//     cuDoubleComplex* field) {
// // Check inputs
// TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
// TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
// TORCH_CHECK(z.device().is_cuda(), "z must be a CUDA tensor");
// TORCH_CHECK(r.device().is_cuda(), "r must be a CUDA tensor");

// TORCH_CHECK(x.dim() == 1, "x must be a 1D tensor");
// TORCH_CHECK(y.dim() == 1, "y must be a 1D tensor");
// TORCH_CHECK(z.dim() == 1, "z must be a 1D tensor");
// TORCH_CHECK(r.dim() == 1, "r must be a 1D tensor");

// int num_particles = x.size(0);
// TORCH_CHECK(y.size(0) == num_particles, "y must have the same size as x");
// TORCH_CHECK(z.size(0) == num_particles, "z must have the same size as x");
// TORCH_CHECK(r.size(0) == num_particles, "r must have the same size as x");


// // Create frequency grids
// // auto host_fx = torch::zeros({N_x}, torch::kDouble);
// // auto host_fy = torch::zeros({N_y}, torch::kDouble);

// double host_fx[N_x];
// double host_fy[N_y];

// // Fill frequency grids
// // for (int i = 0; i < N_x; ++i) {
// //     if (i < N_x / 2) {
// //         host_fx[i] = i / (N_x * dx);
// //     } else {
// //         host_fx[i] = -(N_x - i) / (N_x * dx);
// // }
// // }
// for (int i = 0; i < N_x; ++i) {
//     host_fx[i] = (i < N_x/2) ? i / (N_x * dx) : (i - N_x) / (N_x * dx);
// }

// for (int i = 0; i < N_y; ++i) {
//     host_fy[i] = (i < N_y/2) ? i / (N_y * dy) : (i - N_y) / (N_y * dy);
// }



// double host_kernel_ptr[N_x * N_y];
// double term;
// double arg;
// for (int i = 0; i < N_x; ++i) {
//     for (int j = 0; j < N_y; ++j) {
//         term = 1.0 - lambda * lambda * (host_fx[i] * host_fx[i] + host_fy[j] * host_fy[j]);
//         arg = 2.0 * PI * sqrt(term) / lambda;
//         host_kernel_ptr[i * N_y + j] = arg;
//     }
// }

// double* kernel_ptr;
// cudaMalloc(&kernel_ptr, N_x * N_y * sizeof(double));
// cudaMemcpy(kernel_ptr, host_kernel_ptr, N_x * N_y * sizeof(double), cudaMemcpyHostToDevice);


// // Define grid and block sizes
// dim3 blockSize(16, 16);
// dim3 gridSize((N_x + blockSize.x - 1) / blockSize.x, (N_y + blockSize.y - 1) / blockSize.y);

// // Generate propagation kernel
// // generate_asm_kernel<<<gridSize, blockSize>>>(
// //                 kernel_ptr,
// //                 fx.data_ptr<double>(),
// //                 fy.data_ptr<double>(),
// //                 lambda,
// //                 N_x, N_y);

// // Create cuFFT plan (create once, reuse)
// cufftHandle forward_plan, inverse_plan;
// cufftPlan2d(&forward_plan, N_x, N_y, CUFFT_Z2Z);
// cufftPlan2d(&inverse_plan, N_x, N_y, CUFFT_Z2Z);

// // Process particles in reverse order
// for (int i = num_particles - 1; i >= 0; --i) {
//     // Apply aperture function for current particle
//     set_aperture_kernel<<<gridSize, blockSize>>>(
//     field,
//     x.data_ptr<double>(),
//     y.data_ptr<double>(),
//     r.data_ptr<double>(),
//     i,
//     N_x, N_y
//     );

//     // Check if we need to propagate
//     if (i > 0) {
//         // Get propagation distance
//         double propagation_distance = z[i-1].item<double>() - z[i].item<double>();

//         // // Forward FFT
//         cufftExecZ2Z(forward_plan,
//         field,
//         field,
//         CUFFT_FORWARD);

//         // // Apply propagation kernel
//         apply_propagation_kernel<<<gridSize, blockSize>>>(
//         field,
//         kernel_ptr,
//         propagation_distance,
//         N_x, N_y
//         );

//         // // // Inverse FFT
//         cufftExecZ2Z(inverse_plan,
//         field,
//         field,
//         CUFFT_INVERSE);

//         // // Normalize inverse FFT
//         normalize_fft_kernel<<<gridSize, blockSize>>>(
//         field,
//         N_x, N_y,
//         false);
//         }
//     }

//     // Final propagation to observation plane (based on first particle's z position)
//     double final_propagation = -z[0].item<double>();

//     // Forward FFT
//     cufftExecZ2Z(forward_plan,
//     field,
//     field,
//     CUFFT_FORWARD);

//     // Apply propagation kernel
//     apply_propagation_kernel<<<gridSize, blockSize>>>(
//     field,
//     kernel_ptr,
//     final_propagation,
//     N_x, N_y
//     );

//     // Inverse FFT
//     cufftExecZ2Z(inverse_plan,
//     field,
//     field,
//     CUFFT_INVERSE);

//     // Normalize inverse FFT
//     normalize_fft_kernel<<<gridSize, blockSize>>>(
//     field,
//     N_x, N_y,
//     false
//     );

//     // Clean up
//     cufftDestroy(forward_plan);
//     cufftDestroy(inverse_plan);
//     cudaFree(kernel_ptr);
//     return cudaGetLastError();
// }

#include <cuda_runtime.h>
#include <cuComplex.h> 
#include <math.h>
#include <torch/extension.h>
#include <cufft.h>
#include <stdio.h>

#define EPSILON 2.2204e-16
#define PI 3.14159265358979323846

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

// Kernel for setting aperture (making particles opaque)
__global__ void set_aperture_kernel(cuDoubleComplex* field, 
                                   const double* x, const double* y, const double* r,
                                   int particle_idx, int N_x, int N_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= N_x || idy >= N_y) {
        return;
    }
    
    // Center coordinates at N/2
    double x_cen = (double)(idx - N_x/2);
    double y_cen = (double)(idy - N_y/2);
    
    // Check if point is inside particle
    double rho = (x_cen - x[particle_idx]) * (x_cen - x[particle_idx]) + 
                 (y_cen - y[particle_idx]) * (y_cen - y[particle_idx]);
                 
    if (rho < r[particle_idx] * r[particle_idx]) {
        // If point is inside particle, set to zero (opaque)
        field[idx * N_y + idy] = make_cuDoubleComplex(0.0, 0.0);
    }
}

// Kernel for applying propagation transfer function
__global__ void apply_propagation_kernel(cuDoubleComplex* field, 
                                        const double* kernel, 
                                        double propagation_distance,
                                        int N_x, int N_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= N_x || idy >= N_y) {
        return;
    }
    
    int index = idx * N_y + idy;
        
    double kernel_val = kernel[index] * propagation_distance;
    cuDoubleComplex modified_kernel = make_cuDoubleComplex(cos(kernel_val), sin(kernel_val));
 
    // Multiply field with kernel using complex multiplication
    cuDoubleComplex field_val = field[index];
    cuDoubleComplex result;
    
    result.x = field_val.x * modified_kernel.x - field_val.y * modified_kernel.y;
    result.y = field_val.x * modified_kernel.y + field_val.y * modified_kernel.x;
    
    field[index] = result;
}

// Kernel for normalizing FFT results
__global__ void normalize_fft_kernel(cuDoubleComplex* field, int N_x, int N_y, bool forward) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= N_x || idy >= N_y) {
        return;
    }
    
    int index = idx * N_y + idy;
    
    if (forward) {
        // No normalization needed for forward FFT in cuFFT
        return;
    } else {
        // Normalize inverse FFT by 1/(N_x*N_y)
        double scale = 1.0 / (N_x * N_y);
        field[index].x *= scale;
        field[index].y *= scale;
    }
}

// Generate the angular spectrum propagation kernel directly on device
__global__ void generate_asm_kernel(double* kernel, 
                                   double* fx, double* fy,
                                   double lambda, int N_x, int N_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= N_x || idy >= N_y) {
        return;
    }
    
    int index = idx * N_y + idy;
    
    // Calculate propagation term
    double term = 1.0 - lambda * lambda * (fx[idx] * fx[idx] + fy[idy] * fy[idy]);
    
    // Check for evanescent waves
    if (term < 0) {
        kernel[index] = 0.0; // Evanescent waves
    } else {
        kernel[index] = 2.0 * PI * sqrt(term) / lambda;
    }
}

// Main function for angular spectrum method
cudaError_t asm_propagate(torch::Tensor& x, torch::Tensor& y, torch::Tensor& z,
    torch::Tensor& r, int N_x, int N_y, double lambda, double dx, double dy,
    cuDoubleComplex* field) {
    
    // Check inputs
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(z.device().is_cuda(), "z must be a CUDA tensor");
    TORCH_CHECK(r.device().is_cuda(), "r must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 1, "x must be a 1D tensor");
    TORCH_CHECK(y.dim() == 1, "y must be a 1D tensor");
    TORCH_CHECK(z.dim() == 1, "z must be a 1D tensor");
    TORCH_CHECK(r.dim() == 1, "r must be a 1D tensor");

    int num_particles = x.size(0);
    TORCH_CHECK(y.size(0) == num_particles, "y must have the same size as x");
    TORCH_CHECK(z.size(0) == num_particles, "z must have the same size as x");
    TORCH_CHECK(r.size(0) == num_particles, "r must have the same size as x");

    // Create frequency grids on device
    double *dev_fx, *dev_fy;
    CHECK_CUDA(cudaMalloc(&dev_fx, N_x * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dev_fy, N_y * sizeof(double)));
    
    // Create host frequency grids
    double *host_fx = new double[N_x];
    double *host_fy = new double[N_y];
    
    if (!host_fx || !host_fy) {
        fprintf(stderr, "Failed to allocate host memory for frequency grids\n");
        return cudaErrorMemoryAllocation;
    }
    
    // Fill frequency grids
    for (int i = 0; i < N_x; ++i) {
        host_fx[i] = (i < N_x/2) ? i / (N_x * dx) : (i - N_x) / (N_x * dx);
    }
    
    for (int i = 0; i < N_y; ++i) {
        host_fy[i] = (i < N_y/2) ? i / (N_y * dy) : (i - N_y) / (N_y * dy);
    }
    
    // Copy frequency grids to device
    CHECK_CUDA(cudaMemcpy(dev_fx, host_fx, N_x * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_fy, host_fy, N_y * sizeof(double), cudaMemcpyHostToDevice));
    
    // Free host memory
    delete[] host_fx;
    delete[] host_fy;
    
    // Allocate memory for the kernel
    double* kernel_ptr;
    CHECK_CUDA(cudaMalloc(&kernel_ptr, N_x * N_y * sizeof(double)));
    
    // Optimize block size based on device properties
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    
    int blockSize = 16; // Default
    if (deviceProp.maxThreadsPerBlock >= 1024) {
        blockSize = 32; // Use larger blocks if supported
    }
    
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N_x + blockSize - 1) / blockSize, (N_y + blockSize - 1) / blockSize);
    
    // Generate ASM kernel on device
    generate_asm_kernel<<<gridDim, blockDim>>>(
        kernel_ptr,
        dev_fx,
        dev_fy,
        lambda,
        N_x, N_y
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Free frequency grid memory
    CHECK_CUDA(cudaFree(dev_fx));
    CHECK_CUDA(cudaFree(dev_fy));
    
    // Create cuFFT plans (create once, reuse)
    cufftHandle forward_plan, inverse_plan;
    CHECK_CUFFT(cufftPlan2d(&forward_plan, N_x, N_y, CUFFT_Z2Z));
    CHECK_CUFFT(cufftPlan2d(&inverse_plan, N_x, N_y, CUFFT_Z2Z));
    
    // Process particles in reverse order
    for (int i = num_particles - 1; i >= 0; --i) {
        // Apply aperture function for current particle
        set_aperture_kernel<<<gridDim, blockDim>>>(
            field,
            x.data_ptr<double>(),
            y.data_ptr<double>(),
            r.data_ptr<double>(),
            i,
            N_x, N_y
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Check if we need to propagate
        if (i > 0) {
            // Get propagation distance
            double propagation_distance = z[i-1].item<double>() - z[i].item<double>();
            
            // Forward FFT
            CHECK_CUFFT(cufftExecZ2Z(forward_plan,
                                     field,
                                     field,
                                     CUFFT_FORWARD));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            // Apply propagation kernel
            apply_propagation_kernel<<<gridDim, blockDim>>>(
                field,
                kernel_ptr,
                propagation_distance,
                N_x, N_y
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            
            // Inverse FFT
            CHECK_CUFFT(cufftExecZ2Z(inverse_plan,
                                     field,
                                     field,
                                     CUFFT_INVERSE));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            // Normalize inverse FFT
            normalize_fft_kernel<<<gridDim, blockDim>>>(
                field,
                N_x, N_y,
                false
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
    
    // Final propagation to observation plane (based on first particle's z position)
    double final_propagation = -z[0].item<double>();
    
    // Forward FFT
    CHECK_CUFFT(cufftExecZ2Z(forward_plan,
                            field,
                            field,
                            CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Apply propagation kernel
    apply_propagation_kernel<<<gridDim, blockDim>>>(
        field,
        kernel_ptr,
        final_propagation,
        N_x, N_y
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Inverse FFT
    CHECK_CUFFT(cufftExecZ2Z(inverse_plan,
                            field,
                            field,
                            CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Normalize inverse FFT
    normalize_fft_kernel<<<gridDim, blockDim>>>(
        field,
        N_x, N_y,
        false
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(inverse_plan);
    CHECK_CUDA(cudaFree(kernel_ptr));
    
    return cudaSuccess;
}