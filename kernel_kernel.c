#include "kernel_kernel.h"
/* Module Definition */
void PE(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
{
    int p0 = idx, p1 = idy; // module id
    int local_A[1];
    int local_B[1];
    int local_C[1][1];

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1) {
        // array
        // pe
        local_C[0][0] = 0;
        for (int c2 = 0; c2 <= 127; c2 += 1) {
          // array
          // pe
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            local_A[0] = fifo_A_in.read();
            local_B[0] = fifo_B_in.read();
            local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
            if (c2 == 127 && c5 == 3)
              fifo_C_drain_out.write(local_C[0][0]);
            fifo_B_out.write(local_B[0]);
            fifo_A_out.write(local_A[0]);
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out)
{
    int p0 = idx, p1 = idy; // module id
    int fifo_data;
    int local_A[1][4];

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1)
        for (int c2 = 0; c2 <= 127; c2 += 1) {
          // array
          // io_L3
          for (int c4 = p1; c4 <= 1; c4 += 1) {
            // io_L2
            {
              for (int c6 = 0; c6 <= 3; c6 += 1)
              {
                fifo_data = fifo_A_in.read();
                if (c4 == p1) {
                  local_A[0][c6] = fifo_data;
                } else {
                  fifo_A_out.write(fifo_data);
                }
              }
              if (c4 == p1) {
                // io_L1
                // pe
                for (int c6 = 0; c6 <= 3; c6 += 1)
                  fifo_A_local_out.write(local_A[0][c6]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void A_IO_L3_in(int idx, int *A, hls::stream<int> &fifo_A_local_out)
{
    int p0 = idx; // module id
    int fifo_data;

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1)
        for (int c2 = 0; c2 <= 127; c2 += 1) {
          // array
          // io_L3
          for (int c4 = 0; c4 <= 1; c4 += 1) {
            // io_L2
            for (int c6 = 0; c6 <= 3; c6 += 1)
            {
              fifo_data = A[(2 * p0 + 4 * c0 + c4) * 512 + (4 * c2 + c6)];
              fifo_A_local_out.write(fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in(int idx, int idy, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out)
{
    int p0 = idx, p1 = idy; // module id
    int fifo_data;
    int local_B[4][1];

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1)
        for (int c2 = 0; c2 <= 127; c2 += 1) {
          // array
          // io_L3
          for (int c4 = p1; c4 <= 1; c4 += 1) {
            // io_L2
            {
              for (int c5 = 0; c5 <= 3; c5 += 1)
              {
                fifo_data = fifo_B_in.read();
                if (c4 == p1) {
                  local_B[c5][0] = fifo_data;
                } else {
                  fifo_B_out.write(fifo_data);
                }
              }
              if (c4 == p1) {
                // io_L1
                // pe
                for (int c6 = 0; c6 <= 3; c6 += 1)
                  fifo_B_local_out.write(local_B[c6][0]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_IO_L3_in(int idx, int *B, hls::stream<int> &fifo_B_local_out)
{
    int p0 = idx; // module id
    int fifo_data;

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1)
        for (int c2 = 0; c2 <= 127; c2 += 1) {
          // array
          // io_L3
          for (int c4 = 0; c4 <= 1; c4 += 1) {
            // io_L2
            for (int c5 = 0; c5 <= 3; c5 += 1)
            {
              fifo_data = B[(4 * c2 + c5) * 512 + (2 * p0 + 4 * c1 + c4)];
              fifo_B_local_out.write(fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out(int idx, int idy, int idz, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy, p2 = idz; // module id
    int local_C[1][1];

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1) {
        // array
        // io_L3
        // io_L2
        for (int c5 = 0; c5 <= p2; c5 += 1) {
          // io_L1
          {
            if (c5 == p2) {
              // pe
              local_C[0][0] = fifo_C_drain_local_in.read();
            }
          {
            if (c5 == p2) {
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
void C_drain_IO_L2_out(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1) {
        // array
        // io_L3
        for (int c4 = 0; c4 <= p1; c4 += 1) {
          // io_L2
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // io_L1
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
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L3_out(int idx, int *C, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx; // module id

    for (int c0 = 0; c0 <= 127; c0 += 1)
      for (int c1 = 0; c1 <= 127; c1 += 1) {
        // array
        // io_L3
        for (int c4 = 0; c4 <= 1; c4 += 1) {
          // io_L2
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // io_L1
          {
            fifo_data = fifo_C_drain_local_in.read();
            C[(4 * c0 + c5) * 512 + (2 * p0 + 4 * c1 + c4)] = fifo_data;
          }
          }
        }
      }
}
/* Module Definition */

