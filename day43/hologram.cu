// hologram.cu
#include <cuda_runtime.h>
#include <cuComplex.h> // Provides cuDoubleComplex, cuCadd, cuCmul etc.
#include <math.h>      // Provides math functions like sqrt, cos, sin, j1 for device code
#include <stdio.h>

#define EPSILON 2.2204e-16

// Forward declaration of the kernel
__global__ void hologram_kernel(
    const double* __restrict__ X,
    const double* __restrict__ Y,
    const double* __restrict__ Z,
    const double* __restrict__ R,
    const double* __restrict__ A,
    int num_particles,
    cuDoubleComplex* __restrict__ grid,
    int N_x, int N_y, double dx, double k);

// Note: No need for the __device__ besselj1 wrapper,
// j1(double) is available in CUDA's math.h for device code.

// The actual kernel implementation
__global__ void hologram_kernel(
    const double* __restrict__ X,
    const double* __restrict__ Y,
    const double* __restrict__ Z,
    const double* __restrict__ R,
    const double* __restrict__ A,
    int num_particles,
    cuDoubleComplex* __restrict__ grid,
    int N_x, int N_y, double dx, double k
) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x_idx >= N_x || y_idx >= N_y) return;

    // Calculate grid position (center at N/2)
    int N_x_half = N_x / 2;
    int N_y_half = N_y / 2;
    double x_pos = (double)(x_idx - N_x_half) * dx;
    double y_pos = (double)(y_idx - N_y_half) * dx;

    // Calculate linear index for the grid
    int grid_idx = y_idx * N_x + x_idx;

    // Initialize grid value (background field, typically 1 + 0i)
    cuDoubleComplex val = make_cuDoubleComplex(1.0, 0.0);

    // Accumulate contributions from each particle
    for (int j = 0; j < num_particles; ++j) {
        double dx_ = x_pos - X[j];
        double dy_ = y_pos - Y[j];
        double rho_sq = dx_ * dx_ + dy_ * dy_;
        double rho = sqrt(rho_sq);
        double Zj = Z[j]; // Avoid repeated array access if Z varies per particle

        // Avoid division by zero or near-zero rho
        double denom = rho + EPSILON;

        double Rj = R[j]; // Avoid repeated array access
        double Aj = A[j]; // Avoid repeated array access

        // Calculate arguments for Bessel and exponential functions
        double arg_bessel = k * Rj * rho / Zj;
        // Note: The original formula had k*rho*rho/(2*Z). Check if this is correct.
        // It represents the Fresnel propagation phase factor.
        double arg_exp = k * rho_sq / (2.0 * Zj);

        // Calculate Bessel function J1(x)
        // j1(double) is provided by CUDA's math.h for device code
        cuDoubleComplex bessel_val = make_cuDoubleComplex(j1(arg_bessel), 0.0);

        // Calculate complex exponential e^(i * arg_exp)
        cuDoubleComplex phase = make_cuDoubleComplex(cos(arg_exp), sin(arg_exp));

        // Calculate the scaling factor
        // The original had -A[j]*R[j]/denom * (i/2) = (0 + i*(-A[j]*R[j]/(2*denom)) )
        double scale_imag = Aj * Rj / (2.0 * denom);
        cuDoubleComplex scale_factor = make_cuDoubleComplex(0.0, scale_imag);

        // Calculate the contribution of this particle
        cuDoubleComplex contrib = cuCmul(scale_factor, cuCmul(bessel_val, phase));

        // Add contribution to the grid point value
        val = cuCadd(val, contrib);
    }

    // Write the final complex value to the grid
    grid[grid_idx] = val;
}


// Helper function to check CUDA errors
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        // assert(result == cudaSuccess); // Optional: Halt execution on error
    }
    return result;
}


// C++ wrapper function (Host code) to launch the kernel
// This is what Pybind11 will call. It expects raw pointers.
// We use extern "C" to prevent C++ name mangling, making it easier to link.
extern "C" cudaError_t launch_hologram_kernel(
    const double* d_X, const double* d_Y, const double* d_Z,
    const double* d_R, const double* d_A,
    int num_particles,
    cuDoubleComplex* d_grid, // Note: PyTorch complex tensors map to cuDoubleComplex
    int N_x, int N_y, double dx, double k,
    int threads_x, int threads_y // Pass thread block size from host
) {
    // Define block and grid dimensions
    dim3 threadsPerBlock(threads_x, threads_y);
    dim3 blocksPerGrid(
        (N_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N_y + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    hologram_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_X, d_Y, d_Z, d_R, d_A,
        num_particles,
        d_grid,
        N_x, N_y, dx, k
    );

    // Check for kernel launch errors (asynchronous, so synchronize first for accurate error reporting)
    // checkCuda(cudaGetLastError()); // Optional: Check immediately after launch
    // checkCuda(cudaDeviceSynchronize()); // Ensure kernel completes before returning (important!)

    // Return cudaGetLastError() to let the Pybind wrapper check kernel status
    return cudaGetLastError();
}