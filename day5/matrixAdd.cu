#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixAdd(int *a, int *b, int *c, int m, int n){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    // tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx < m*n){
        c[tidx] = a[tidx] + b[tidx];
    }
}


int main(){
    int m,n;
    printf("Enter num rows:\n");
    scanf("%d",&m);
    printf("Enter num cols:\n");
    scanf("%d",&n);

    int a[m][n];
    int b[m][n];
    // int c[m][n];

    printf("Enter matrix A..");
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            scanf("%d",&a[i][j]);
        }
    }
    printf("Enter matrix B...");
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            scanf("%d",&b[i][j]);
        }
    }

    // flatten A and B 
    int fa[m*n];
    int fb[m*n];
    int fc[m*n];

    int k = 0;
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            fa[k] = a[i][j];
            k++;            
        }
    }
    k=0;
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            fb[k] = b[i][j];
            k++ ;           
        }
    }

    // intialize arrays (pointers) for gpu
    int *ga;
    int *gb;
    int *gc;

    //Allocate memory for above pointers/arrays in gpu
    cudaMalloc((void**)&ga, m*n*sizeof(int));
    cudaMalloc((void**)&gb, m*n*sizeof(int));
    cudaMalloc((void**)&gc, m*n*sizeof(int));

    //copy matrices to gpu
    cudaMemcpy(ga, fa, m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gb, fb, m*n*sizeof(int), cudaMemcpyHostToDevice);

    int numThreads = m*n;
    int numBlocks = 1;
    
    // Assign grids, blocks and threads
    // dim3 blockDim(4,4);
    // dim3 gridDim((1,1));
    
    // launch kernel to add 
    matrixAdd <<< numBlocks, numThreads >>> (ga,gb,gc,m,n);

    // copy result to c
    cudaMemcpy(fc, gc, m*n*sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gb);

    // print in matrix form
    k = 0;
    for (int i=0; i<m; ++i){
        for (int j=0;j<n;++j){
            printf(" %d ",fc[k]);
            if (k%n==1){
                printf("\n");
            }
            k++;
        }

    }

    return 0;

}