// CUDA code for angular spectrum method (ASM) for holography

#include <cuda_runtime.h>
#include <cuComplex.h> 
#include <math.h>
#include <cufft.h>
#include <iostream>
#include <torch/extension.h>

# define EPSILON 2.2204e-16
# define PI 3.14159265358979323846


// fft->pointwise->ifft
// user will input list of particles {(x_j,y_j,z_j,r_j), j = 1,..,N}
__global__ void asm_kernel(double *x, double* y, double* z, double* r,
    int num_particles, cuDoubleComplex* kernel, cuDoubleComplex* grid,  int N_x, int N_y){
        int idx = threadIdx.x+blockIdx.x*blockDim.x;
        int idy = threadIdx.y+blockIdx.y*blockDim.y;
        
        if (idx >=N_x || idy >=N_y) {
            return;
        }
        
        // treat each thread as a grid point, center them at N/2
        double x_cen = (double)(idx - N_x/2);
        double y_cen = (double)(idy - N_y/2);

        
        // need to find indices for each particle which are to be made 0
        for (int i = 0; i < num_particles-1; ++i){
            double rho = (x_cen-x[i])*(x_cen-x[i]) + (y_cen-y[i])*(y_cen-y[i]);
            if (rho < r[i]*r[i]){
                // if the particle is in the grid, set the grid point to 0
                // this is the pointwise multiplication
                grid[idx*N_y+idy] = make_cuDoubleComplex(0.0,0.0);
            }

            __syncthreads();
              // fft of the grid
            cufftHandle plan;
            cufftPlan2d(&plan, N_x, N_y,CUFFT_Z2Z);
            cufftExecZ2Z(plan, grid, grid, CUFFT_FORWARD);
            cufftDestroy(plan);

            // poinwise multiplication
            __syncthreads();
            grid[idx*N_y+idy] = cuCmul(grid[idx*N_y+idy], cuCmul(z[i+1]-z[i],kernel[idx*N_y+idy]));

            __syncthreads();
            // ifft of the grid
            cufftPlan2d(&plan, N_x, N_y,CUFFT_Z2Z);         
            cufftExecZ2Z(plan, grid, grid, CUFFT_INVERSE);
            cufftDestroy(plan);
            
            // normalize the grid
            __syncthreads();
            grid[idx*N_y+idy] = make_cuDoubleComplex(1.0/(N_x*N_y),0.0);
        
        }
        // copy the grid to the output
        // grid[idx*N_y+idy] = make_cuDoubleComplex(0.0,0.0);

    }   




void asm(torch::Tensor& x, torch::Tensor& y, torch::Tensor& z,
    torch::Tensor& r, int N_x, int N_y, double lambda);

void asm(torch::Tensor& x, torch::Tensor& y, torch::Tensor& z,
    torch::Tensor& r, int N_x, int N_y, double lambda, double dx){
        int num_particles = x.size(0);
        double fx[N_x], fy[N_y];
        cuDoubleComplex grid[N_x][N_y];
        // Initialize grid to 1
        for (int i=0; i <N_x; ++i){
            for (int j=0; j<N_y; ++j){
                grid[i][j] = make_cuDoubleComplex(1.0,0);
            }
        }


        for (int i=0; i <N_x; ++i){
            if (i < N_x/2){
                fx[i] = i/(N_x*dx);
            }
            if (i >=N_x/2){
                fx[i]=-(N_x-i)/(N_x*dx);
            }
        }
        for (int i=0; i <N_y; ++i){
            if (i < N_y/2){
                fy[i] = i/(N_x*dx);
            }
            if (i >=N_y/2){
                fy[i]=-(N_y-i)/(N_y*dx);
            }
        }
        double arg;
        cuDoubleComplex kernel[N_x][N_y]; 
        for (int i=0; i<N_x; ++i){
            for (int j=0;j<N_y;++j){
                arg = 2*PI*sqrt(1-lambda*lambda*fx[i]*fx[i]+fy[j]*fy[j])/(lambda);
                kernel[i][j] = make_cuDoubleComplex(cos(arg), sin(arg)); 
            }
        }

        // Allocate device memory 
        double *d_x, *d_y, *d_z, *d_r;
        cuDoubleComplex *d_kernel, *d_grid;

        cudaMalloc((void**)&d_x, num_particles*sizeof(double));
        cudaMalloc((void**)&d_y, num_particles*sizeof(double));
        cudaMalloc((void**)&d_z, num_particles*sizeof(double));
        cudaMalloc((void**)&d_r, num_particles*sizeof(double));
        cudaMalloc((void**)&d_kernel, N_x*N_y*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_grid, N_x*N_y*sizeof(cuDoubleComplex));

        // copy data to device
        cudaMemcpy(d_x, x.data_ptr<double>(), num_particles*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y.data_ptr<double>(), num_particles*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z.data_ptr<double>(), num_particles*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r.data_ptr<double>(), num_particles*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, N_x*N_y*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grid, grid, N_x*N_y*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Define grid and block sizes
        dim3 blockSize(16,16);
        dim3 gridSize((N_x + blockSize.x - 1)/blockSize.x, (N_y+blockSize.y -1)/blockSize.y);

        //Launch kernel;
        asm_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_r, num_particles, d_kernel, d_grid, N_x, N_y);

        // No need to copy back the kernel, we will do it in the pytorch code in the end
    


    }







