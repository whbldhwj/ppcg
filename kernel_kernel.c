#include "kernel_kernel.h"
/* Module Definition */
void PE(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
{
    int p0 = idx, p1 = idy; // module id
    int local_A[1];
    int local_B[1];
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          // pe
          {
            if (c2 == 0)
              local_C[0][0] = 0;
            for (int c5 = 0; c5 <= 1; c5 += 1) {
              local_A[0] = fifo_A_in.read();
              local_B[0] = fifo_B_in.read();
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              fifo_B_out.write(local_B[0]);
              fifo_A_out.write(local_A[0]);
            }
            if (c2 == 255)
              fifo_C_drain_out.write(local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
void A_L2_in_IO(int idx, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out)
{
    int p0 = idx; // module id
    int fifo_data;
    int local_A[1][2];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          // io_L2
          for (int c3 = p0; c3 <= 1; c3 += 1) {
            for (int c5 = 0; c5 <= 1; c5 += 1)
            {
              fifo_data = fifo_A_in.read();
              if (c3 == p0) {
                local_A[0][c5] = fifo_data;
              } else {
                fifo_A_out.write(fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  fifo_A_local_out.write(local_A[0][c5]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void A_L3_in_IO(int *A, hls::stream<int> &fifo_A_local_out)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          // io_L2
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L1
            for (int c5 = 0; c5 <= 1; c5 += 1)
            {
              fifo_data = A[(2 * c0 + c3) * 512 + (2 * c2 + c5)];
              fifo_A_local_out.write(fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_L2_in_IO(int idx, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out)
{
    int p0 = idx; // module id
    int fifo_data;
    int local_B[2][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          // io_L2
          for (int c3 = p0; c3 <= 1; c3 += 1) {
            for (int c4 = 0; c4 <= 1; c4 += 1)
            {
              fifo_data = fifo_B_in.read();
              if (c3 == p0) {
                local_B[c4][0] = fifo_data;
              } else {
                fifo_B_out.write(fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  fifo_B_local_out.write(local_B[c5][0]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_L3_in_IO(int *B, hls::stream<int> &fifo_B_local_out)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          // io_L2
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1)
            {
              fifo_data = B[(2 * c2 + c4) * 512 + (2 * c1 + c3)];
              fifo_B_local_out.write(fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void C_drain_L1_out_IO(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {
            // array
            // io_L2
            // io_L1
            for (int c4 = 0; c4 <= p1; c4 += 1) {
              // pe
              {
                local_C[0][0] = fifo_C_drain_local_in.read();
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = fifo_C_drain_in.read();
                }
                fifo_C_drain_out.write(fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
void C_drain_L2_out_IO(int idx, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx; // module id
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {
            // array
            // io_L2
            for (int c3 = 0; c3 <= p0; c3 += 1) {
              // io_L1
              for (int c4 = 0; c4 <= 1; c4 += 1) {
                // pe
              {
                if (c3 == p0) {
                  fifo_data = fifo_C_drain_in.read();
                } else {
                  fifo_data = fifo_C_drain_in.read();
                }
                fifo_C_drain_out.write(fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
void C_drain_L3_out_IO(int *C, hls::stream<int> &fifo_C_drain_local_in)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {
            // array
            // io_L2
            for (int c3 = 0; c3 <= 1; c3 += 1) {
              // io_L1
              for (int c4 = 0; c4 <= 1; c4 += 1) {
                // pe
              {
                fifo_data = fifo_C_drain_local_in.read();
                C[(2 * c0 + c4) * 512 + (2 * c1 + c3)] = fifo_data;
              }
              }
            }
          }
}
/* Module Definition */

