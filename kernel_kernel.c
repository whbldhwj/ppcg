#include "kernel_kernel.h"
void kernel0(int *A, int *B, int *C)
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
                if (c2 >= 1) {
                  local_C[0][0] = C[(8 * c0 + c3) * 512 + (8 * c1 + c4)];
                } else {
                  local_C[0][0] = 0;
                }
                for (int c5 = 0; c5 <= 7; c5 += 1)
                  local_C[0][0] = (local_C[0][0] + (A[(8 * c0 + c3) * 512 + (8 * c2 + c5)] * B[(8 * c2 + c5) * 512 + (8 * c1 + c4)]));
                C[(8 * c0 + c3) * 512 + (8 * c1 + c4)] = local_C[0][0];
              }
            }
        }
}

void C_drain_L1_out_IO(hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in, int idx, int idy)
{
    int p0 = idx, p1 = idy;
    int fifo_data;

    for (int c0 = 0; c0 <= 63; c0 += 1)
      for (int c1 = 0; c1 <= 63; c1 += 1)
        for (int c2 = 0; c2 <= 63; c2 += 1)
          if (c2 == 63) {
            // array
            // io_L2
            // io_L1
            for (int c4 = 0; c4 <= p1; c4 += 1) {
              // pe
            {
              if (c4 == p1) {
                fifo_data = fifo_C_drain_local_in.read();
              } else {
                fifo_data = fifo_C_drain_in.read();
              }
              fifo_C_drain_out.write(fifo_data);
            }
            }
          }
}

