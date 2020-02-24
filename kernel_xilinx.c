#include "kernel_kernel.h"
/* CPU Kernel Definition */
void kernel0(int *A, int *B, int *C)
{
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {
          // array
          for (int c3 = 0; c3 <= 1; c3 += 1)
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                if (c2 >= 1) {
                  local_C[0][0] = C[(2 * c0 + c3) * 512 + (2 * c1 + c4)];
                } else {
                  local_C[0][0] = 0;
                }
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  local_C[0][0] = (local_C[0][0] + (A[(2 * c0 + c3) * 512 + (2 * c2 + c5)] * B[(2 * c2 + c5) * 512 + (2 * c1 + c4)]));
                C[(2 * c0 + c3) * 512 + (2 * c1 + c4)] = local_C[0][0];
              }
            }
        }
}
/* CPU Kernel Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void PE(int idx, int idy, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_C_drain_out)
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
              local_A[0] = read_channel_intel(fifo_A_in);
              local_B[0] = read_channel_intel(fifo_B_in);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_out, local_B[0]);
              write_channel_intel(fifo_A_out, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_out, local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void A_L2_in_IO(int idx, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_A_local_out)
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
              fifo_data = read_channel_intel(fifo_A_in);
              if (c3 == p0) {
                local_A[0][c5] = fifo_data;
              } else {
                write_channel_intel(fifo_A_out, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_A_local_out, local_A[0][c5]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void A_L3_in_IO(int *A, channel int fifo_A_local_out)
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
              write_channel_intel(fifo_A_local_out, fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void B_L2_in_IO(int idx, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_B_local_out)
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
              fifo_data = read_channel_intel(fifo_B_in);
              if (c3 == p0) {
                local_B[c4][0] = fifo_data;
              } else {
                write_channel_intel(fifo_B_out, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_B_local_out, local_B[c5][0]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void B_L3_in_IO(int *B, channel int fifo_B_local_out)
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
              write_channel_intel(fifo_B_local_out, fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO(int idx, int idy, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in)
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
                local_C[0][0] = read_channel_intel(fifo_C_drain_local_in);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                }
                write_channel_intel(fifo_C_drain_out, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L2_out_IO(int idx, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in)
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
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                }
                write_channel_intel(fifo_C_drain_out, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L3_out_IO(int *C, channel int fifo_C_drain_local_in)
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
                fifo_data = read_channel_intel(fifo_C_drain_local_in);
                C[(2 * c0 + c4) * 512 + (2 * c1 + c3)] = fifo_data;
              }
              }
            }
          }
}
/* Module Definition */

void kernel0(int *A, int *B, int *C)
{
    /* FIFO Declaration */
    /* PE fifo */ channel int fifo_A_PE_0_0;
    /* PE fifo */ channel int fifo_A_PE_0_1;
    /* PE fifo */ channel int fifo_A_PE_0_2;
    /* PE fifo */ channel int fifo_A_PE_1_0;
    /* PE fifo */ channel int fifo_A_PE_1_1;
    /* PE fifo */ channel int fifo_A_PE_1_2;
    /* PE fifo */ channel int fifo_B_PE_0_0;
    /* PE fifo */ channel int fifo_B_PE_1_0;
    /* PE fifo */ channel int fifo_B_PE_2_0;
    /* PE fifo */ channel int fifo_B_PE_0_1;
    /* PE fifo */ channel int fifo_B_PE_1_1;
    /* PE fifo */ channel int fifo_B_PE_2_1;
    /* PE fifo */ channel int fifo_C_drain_PE_0_0;
    /* PE fifo */ channel int fifo_C_drain_PE_1_0;
    /* PE fifo */ channel int fifo_C_drain_PE_0_1;
    /* PE fifo */ channel int fifo_C_drain_PE_1_1;
    /* A_L2_in_IO fifo */ channel int fifo_A_A_L2_in_IO_0;
    /* A_L2_in_IO fifo */ channel int fifo_A_A_L2_in_IO_1;
    /* A_L2_in_IO fifo */ channel int fifo_A_A_L2_in_IO_2;
    /* B_L2_in_IO fifo */ channel int fifo_B_B_L2_in_IO_0;
    /* B_L2_in_IO fifo */ channel int fifo_B_B_L2_in_IO_1;
    /* B_L2_in_IO fifo */ channel int fifo_B_B_L2_in_IO_2;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_0_0;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_0_1;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_0_2;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_1_0;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_1_1;
    /* C_drain_L1_out_IO fifo */ channel int fifo_C_drain_C_drain_L1_out_IO_1_2;
    /* C_drain_L2_out_IO fifo */ channel int fifo_C_drain_C_drain_L2_out_IO_0;
    /* C_drain_L2_out_IO fifo */ channel int fifo_C_drain_C_drain_L2_out_IO_1;
    /* C_drain_L2_out_IO fifo */ channel int fifo_C_drain_C_drain_L2_out_IO_2;
    /* FIFO Declaration */

    /* Module Call */
    PE(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_A_PE_0_0,
        /* fifo */ fifo_A_PE_0_1,
        /* fifo */ fifo_B_PE_0_0,
        /* fifo */ fifo_B_PE_1_0,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_A_PE_0_1,
        /* fifo */ fifo_A_PE_0_2,
        /* fifo */ fifo_B_PE_0_1,
        /* fifo */ fifo_B_PE_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_A_PE_1_0,
        /* fifo */ fifo_A_PE_1_1,
        /* fifo */ fifo_B_PE_1_0,
        /* fifo */ fifo_B_PE_2_0,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_A_PE_1_1,
        /* fifo */ fifo_A_PE_1_2,
        /* fifo */ fifo_B_PE_1_1,
        /* fifo */ fifo_B_PE_2_1,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    A_L2_in_IO(
        /* module id */ 0,
        /* fifo */ fifo_A_A_L2_in_IO_0,
        /* fifo */ fifo_A_A_L2_in_IO_1,
        /* fifo */ fifo_A_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    A_L2_in_IO(
        /* module id */ 1,
        /* fifo */ fifo_A_A_L2_in_IO_1,
        /* fifo */ fifo_A_A_L2_in_IO_2,
        /* fifo */ fifo_A_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    A_L3_in_IO(
        /* array */ A,
        /* fifo */ fifo_A_A_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    B_L2_in_IO(
        /* module id */ 0,
        /* fifo */ fifo_B_B_L2_in_IO_0,
        /* fifo */ fifo_B_B_L2_in_IO_1,
        /* fifo */ fifo_B_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    B_L2_in_IO(
        /* module id */ 1,
        /* fifo */ fifo_B_B_L2_in_IO_1,
        /* fifo */ fifo_B_B_L2_in_IO_2,
        /* fifo */ fifo_B_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    B_L3_in_IO(
        /* array */ B,
        /* fifo */ fifo_B_B_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_1,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_2,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_2,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_L2_out_IO(
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_0,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_L2_out_IO(
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_1,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_2,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_L3_out_IO(
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_2
    );
    /* Module Call */

}
