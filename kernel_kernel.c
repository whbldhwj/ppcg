#include "kernel_kernel.h"
__global__ void kernel0(int *A, int *B, int *C)
{
    int local_A[8][8];
    int local_B[8][8];
    int local_C[8][8];

    for (int c0 = 0; c0 <= 63; c0 += 1)
      for (int c1 = 0; c1 <= 63; c1 += 1) {
        for (int c2 = 0; c2 <= 63; c2 += 1) {
          for (int c3 = 0; c3 <= 7; c3 += 1)
            for (int c4 = 0; c4 <= 7; c4 += 1)
              local_A[c3][c4] = A[(8 * c0 + c3) * 512 + (8 * c2 + c4)];
          for (int c3 = 0; c3 <= 7; c3 += 1)
            for (int c4 = 0; c4 <= 7; c4 += 1)
              local_B[c3][c4] = B[(8 * c2 + c3) * 512 + (8 * c1 + c4)];
          // array
          for (int c3 = 0; c3 <= 7; c3 += 1)
            for (int c4 = 0; c4 <= 7; c4 += 1) {
              if (c2 == 0)
                local_C[c3][c4] = 0;
              for (int c5 = 0; c5 <= 7; c5 += 1)
                local_C[c3][c4] = (local_C[c3][c4] + (local_A[c3][c5] * local_B[c5][c4]));
            }
        }
        for (int c2 = 0; c2 <= 7; c2 += 1)
          for (int c3 = 0; c3 <= 7; c3 += 1)
            C[(8 * c0 + c2) * 512 + (8 * c1 + c3)] = local_C[c2][c3];
      }
}
