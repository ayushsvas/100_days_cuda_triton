#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>

__global__ void transpose(float*a, float*b, int col_size, int row_size){
        int tidx = threadIdx.x + threadIdx.y * blockDim.x;
        int tidy = threadIdx.y + threadIdx.x * blockDim.y;
       
        if (tidx>=col_size*row_size || tidy >= col_size*row_size) return;
       
        b[tidx] = a[tidy];

}


int main(){
        const int col_size = 4;
        float arr[][col_size] = {{1.0,2.0,3.1,4.1}, {1.0,2.4,3.5,4.7}, {1.2,2.2,3.2,4.2}, {1.1,2.1,3.1,4.1}};

        const int row_size = sizeof(arr) / (col_size*sizeof(float));
        std:: cout<<row_size<<" "<<col_size<<"\n";
        
        float *da, *db;
        cudaMalloc(&da, col_size*row_size*sizeof(float));
        cudaMalloc(&db, col_size*row_size*sizeof(float));
    
        cudaMemcpy(da, arr, col_size*row_size*sizeof(float), cudaMemcpyHostToDevice);

        dim3 numThreadsPerBlock(col_size,row_size);
    
        transpose <<< 1, numThreadsPerBlock >>> (da,db,col_size,row_size);

        cudaMemcpy(arr, db, col_size*row_size*sizeof(float), cudaMemcpyDeviceToHost);

        for (int i=0;i<col_size;++i){
            for (int j=0;j<row_size;++j){
                std:: cout<<arr[i][j]<<" "; 
            }
            std:: cout<<"\n";
        }   

        cudaFree(da);
        cudaFree(db);

        return 0;



}

