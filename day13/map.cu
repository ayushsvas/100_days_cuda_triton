#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void map_fn(int *a, int *b, int n){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < n){
        b[tidx] = a[tidx] + 10;
    }
}

int main(){
    int a[10];
    int o[10];
    printf("Enter 10 digits...\n");
    for (int i=0;i<10;++i){
        scanf("%d", &a[i]); 
    }

    int *g_a;
    int *g_o;

    cudaMalloc((void**)&g_a, 10*sizeof(int));
    cudaMalloc((void**)&g_o, 10*sizeof(int));

    cudaMemcpy(g_a, a, 10*sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 10;
    int blocksPerGrid = 10;

    map_fn <<< blocksPerGrid,threadsPerBlock >>> (g_a, g_o, 10);

    cudaMemcpy(o, g_o, 10*sizeof(int), cudaMemcpyDeviceToHost);

    printf("After adding...\n");
    for (int i=0;i<10;++i){
        printf("%d\n", o[i]);
    }

    cudaFree(g_a);
    cudaFree(g_o);

    return 0;

    
}