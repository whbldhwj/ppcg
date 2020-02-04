#include <assert.h>
#include <stdio,h>
#include "kernel_kernel.h"
#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

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
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_C);
      cudaCheckKernel();
    }
    
    cudaCheckReturn(cudaMemcpy(C, dev_C, (512) * (512) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_B));
    cudaCheckReturn(cudaFree(dev_C));
  }
  // kernel
