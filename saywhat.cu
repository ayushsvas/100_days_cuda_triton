#include <cuda_runtime.h>
#include <string> // to read multiple words
#include <cstdlib> //malloc
#include <iostream> 
#include <cstring> 

__device__ char to_upper(char c){ // cuda doesn't toupper and tolower 
    if (c >= 'a' && c <= 'z'){
        return c - 32;
    }
    return c;
}

__global__ void saywhat(char* a, char* b, int strlen){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid > strlen) return;
    b[tid] = to_upper(a[tid]);
}

int main(){
    std::string input;
    std::cout << "Enter a string: ";
    std::getline(std::cin, input); //cin reads single words 

    char* ha = (char*)malloc(input.length()+1);
    std::strcpy(ha, input.c_str());


    char *a, *b;
    cudaMalloc(&a, (input.length()+1)*sizeof(char));
    cudaMalloc(&b, (input.length()+1)*sizeof(char));
    cudaMemcpy(a, ha, (input.length()+1)*sizeof(char), cudaMemcpyHostToDevice);
    

    int numThreadsPerBlock = 16;
    int numBlocksInGrid = (input.length()+1+numThreadsPerBlock-1)/numThreadsPerBlock;
    saywhat <<< numBlocksInGrid, numThreadsPerBlock >>> (a, b, input.length()+1);
    
    std::cout <<"Shouting... \n";
    cudaMemcpy(ha, b, (input.length()+1)*sizeof(char), cudaMemcpyDeviceToHost);
    std::cout<< ha <<std::endl;
    // std::cout <<"\n";

    cudaFree(a);
    cudaFree(b);
    free(ha);

}