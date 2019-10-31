#include "kernel_kernel.hu"
__global__ void kernel0(int *A, int *B, int *C)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ int shared_A[32][32];
    __shared__ int shared_B[32][32];
    int private_C[1][2];

    {
      for (int c2 = 0; c2 <= 511; c2 += 32) {
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_A[t0][c4] = A[(32 * b0 + t0) * 512 + (c2 + c4)];
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_B[t0][c4] = B[(t0 + c2) * 512 + (32 * b1 + c4)];
        __syncthreads();
        if (c2 == 0) {
          private_C[0][0] = 0;
          private_C[0][1] = 0;
        }
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_C[0][0] += (shared_A[t0][c3] * shared_B[c3][t1]);
          private_C[0][1] += (shared_A[t0][c3] * shared_B[c3][t1 + 16]);
        }
        __syncthreads();
      }
      C[(32 * b0 + t0) * 512 + (32 * b1 + t1)] = private_C[0][0];
      C[(32 * b0 + t0) * 512 + (32 * b1 + t1 + 16)] = private_C[0][1];
    }
}
