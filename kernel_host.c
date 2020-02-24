#include <assert.h>
#include <stdio,h>
#include "kernel_kernel.h"
#include "kernel_top_gen.h"
#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  {
    int *dev_A;
    int *dev_B;
    int *dev_C;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (512) * (512) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_B, (512) * (512) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_C, (512) * (512) * sizeof(int)));
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (512) * (512) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_B, B, (512) * (512) * sizeof(int), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(1);
      dim3 k0_dimGrid(1);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B, dev_C);
      cudaCheckKernel();
      /* Top Function Generation */
      FILE *f = fopen("top.c", "w");
      top_generate(f);
      fclose(f);
      /* Top Function Generation */
    }
    
    cudaCheckReturn(cudaMemcpy(C, dev_C, (512) * (512) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_B));
    cudaCheckReturn(cudaFree(dev_C));
  }
}
