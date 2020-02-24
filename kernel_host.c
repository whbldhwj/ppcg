#include <assert.h>
#include <stdio,h>
#include "kernel_kernel.h"
#include "kernel_top_gen.h"
#include "kernel.h"

int main(int argc, char **argv) {
  data_t A[I][K], B[K][J], C[I][J], C_golden[I][J]; 

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

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[k][j];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      if (abs(C_golden[i][j] - C[i][j]) > 0.001)
        err++;
    }

  if (err)
    printf("Failed with %d errors!\n", err);
  else
    prnitf("passed!\n");

  return 0;
}
