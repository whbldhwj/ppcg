#include "kernel_kernel.h"
__global__ void kernel0(int *A, int *B, int *C)
{
    int local_C[1][1];

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

void PE(hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_in, hls::stream<int> &fifo_C_out, hls::stream<int> &fifo_C_drain_out, int idx, int idy)
{
    int p0 = idx, p1 = idy;
    int local_A[1];
    int local_B[1];
    int local_C[1][1];

    for (int c0 = 0; c0 <= 63; c0 += 1)
      for (int c1 = 0; c1 <= 63; c1 += 1)
        for (int c2 = 0; c2 <= 63; c2 += 1) {
          // array
          // pe
          {
            if (c2 >= 1) {
              local_C[0][0] = fifo_C_in.read();
            } else {
              local_C[0][0] = 0;
            }
            for (int c5 = 0; c5 <= 7; c5 += 1) {
              local_A[0] = fifo_A_in.read();
              local_B[0] = fifo_B_in.read();
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              fifo_B_out.write(local_B[0];
              fifo_A_out.write(local_A[0];
            }
            if (c2 <= 62) {
              fifo_C_out.write(local_C[0][0];
            } else {
              fifo_C_drain_out.write(local_C[0][0];
            }
          }
        }
}
