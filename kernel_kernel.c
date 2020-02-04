#include "kernel_kernel.h"
__global__ void kernel0(int *C)
{

    for (int c0 = 0; c0 <= 63; c0 += 1)
      for (int c1 = 0; c1 <= 63; c1 += 1)
        for (int c2 = 0; c2 <= 63; c2 += 1) {
          // array
          for (int c3 = 0; c3 <= 7; c3 += 1)
            for (int c4 = 0; c4 <= 7; c4 += 1) {
              // pe
              {
                local_C[0][0] = C[(8 * c0 + c3) * 512 + (8 * c1 + c4)];
                if (c2 == 0)
                  local_C[0][0] = 0;
                for (int c5 = 0; c5 <= 7; c5 += 1)
                  local_C[0][0] = (local_C[0][0] + (A[(8 * c0 + c3) * 512 + (8 * c2 + c5)] * B[(8 * c2 + c5) * 512 + (8 * c1 + c4)]));
                C[(8 * c0 + c3) * 512 + (8 * c1 + c4)] = local_C[0][0];
              }
            }
        }
}
{

    for (int c0 = 0; c0 <= 63; c0 += 1)
      for (int c1 = 0; c1 <= 63; c1 += 1)
        for (int c2 = 0; c2 <= 63; c2 += 1) {
          // array
          {
            // pe
            {
              local_C[0][0] = fifo_C.read();
              if (c2 == 0)
                local_C[0][0] = 0;
              for (int c5 = 0; c5 <= 7; c5 += 1) {
                local_A[0] = fifo_A.read();
                local_B[0] = fifo_B.read();
                local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
                fifo_B.write(local_B[0];
                fifo_A.write(local_A[0];
              }
              if (c2 <= 62)
                fifo_C.write(local_C[0][0];
            }
            if (c2 == 63)
              fifo_C_drain.write(local_C[0][0];
          }
        }
}
