#include <stdio.h>
#include <cuda_runtime.h>

__global__ void broadcast(int *a, int *b, int *c, int m, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m*n){
        c[tid] = a[tid]+b[tid];
    }
}

int main(){
    int m,n;
    printf("Enter vector A length:");
    scanf("%d", &m);

    printf("Enter vector B length:");
    scanf("%d", &n);

    int a[m];
    int b[n];
    printf("Enter vector A:\n");
    for (int i=0; i<m; ++i){
        scanf("%d", &a[i]);
    }
    
    printf("Enter vector B:\n");
    for (int i=0; i<m; ++i){
        scanf("%d", &b[i]);
    }

    // adding and broadcasting. Dupicate same A and transpose B

    int ba[m*n];
    int bb[m*n];
    int bc[m*n];
    for (int i=0; i<m; ++i){
        for (int j=0; j<n; ++j)
            ba[i*n+j] = a[i];
    }

    for (int i=0; i<m; ++i){
        for (int j=0; j<n; ++j)
                bb[i*n+j] = b[j];                

    }

    // assign pointers to gpu memory and allocate memeory
    int *ga, *gb, *gc;

    cudaMalloc((void **)&ga, m*n*sizeof(int));
    cudaMalloc((void **)&gb, m*n*sizeof(int));
    cudaMalloc((void **)&gc, m*n*sizeof(int));

    // copy data to device
    cudaMemcpy(ga, ba, m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gb, bb, m*n*sizeof(int), cudaMemcpyHostToDevice);

    int numThreads = m*n;
    int numBlocks = 1;

    broadcast <<< numBlocks, numThreads >>> (ga, gb, gc, m, n);

    // copy result to host device
    cudaMemcpy(bc, gc, m*n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);

    // print result in matrix for 
    for (int i = 0; i<m*n; ++i){
        printf(" %d ", bc[i]);
        if (i%n==1){
            printf("\n");
        }
    }

    return 0;

}