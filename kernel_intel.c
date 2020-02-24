#include "kernel_kernel.h"

/* Channel Declaration */
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
/* Channel Declaration */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void PE()
{
    int p0 = 0, p1 = 0; // module id
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
              local_A[0] = read_channel_intel(fifo_A_PE_0_0);
              local_B[0] = read_channel_intel(fifo_B_PE_0_0);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_PE_1_0, local_B[0]);
              write_channel_intel(fifo_A_PE_0_1, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_PE_0_0, local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void PE()
{
    int p0 = 0, p1 = 1; // module id
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
              local_A[0] = read_channel_intel(fifo_A_PE_0_1);
              local_B[0] = read_channel_intel(fifo_B_PE_0_1);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_PE_1_1, local_B[0]);
              write_channel_intel(fifo_A_PE_0_2, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_PE_0_1, local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void PE()
{
    int p0 = 1, p1 = 0; // module id
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
              local_A[0] = read_channel_intel(fifo_A_PE_1_0);
              local_B[0] = read_channel_intel(fifo_B_PE_1_0);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_PE_2_0, local_B[0]);
              write_channel_intel(fifo_A_PE_1_1, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_PE_1_0, local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void PE()
{
    int p0 = 1, p1 = 1; // module id
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
              local_A[0] = read_channel_intel(fifo_A_PE_1_1);
              local_B[0] = read_channel_intel(fifo_B_PE_1_1);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_PE_2_1, local_B[0]);
              write_channel_intel(fifo_A_PE_1_2, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_PE_1_1, local_C[0][0]);
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void A_L2_in_IO()
{
    int p0 = 0; // module id
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
              fifo_data = read_channel_intel(fifo_A_A_L2_in_IO_0);
              if (c3 == p0) {
                local_A[0][c5] = fifo_data;
              } else {
                write_channel_intel(fifo_A_A_L2_in_IO_1, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_A_PE_0_0, local_A[0][c5]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void A_L2_in_IO()
{
    int p0 = 1; // module id
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
              fifo_data = read_channel_intel(fifo_A_A_L2_in_IO_1);
              if (c3 == p0) {
                local_A[0][c5] = fifo_data;
              } else {
                write_channel_intel(fifo_A_A_L2_in_IO_2, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_A_PE_1_0, local_A[0][c5]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void A_L3_in_IO(int *A)
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
              write_channel_intel(fifo_A_A_L2_in_IO_0, fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void B_L2_in_IO()
{
    int p0 = 0; // module id
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
              fifo_data = read_channel_intel(fifo_B_B_L2_in_IO_0);
              if (c3 == p0) {
                local_B[c4][0] = fifo_data;
              } else {
                write_channel_intel(fifo_B_B_L2_in_IO_1, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_B_PE_0_0, local_B[c5][0]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void B_L2_in_IO()
{
    int p0 = 1; // module id
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
              fifo_data = read_channel_intel(fifo_B_B_L2_in_IO_1);
              if (c3 == p0) {
                local_B[c4][0] = fifo_data;
              } else {
                write_channel_intel(fifo_B_B_L2_in_IO_2, fifo_data);
              }
            }
            // io_L1
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              // pe
              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_B_PE_0_1, local_B[c5][0]);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void B_L3_in_IO(int *B)
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
              write_channel_intel(fifo_B_B_L2_in_IO_0, fifo_data);
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO()
{
    int p0 = 0, p1 = 0; // module id
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
                local_C[0][0] = read_channel_intel(fifo_C_drain_PE_0_0);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L1_out_IO_0_0);
                }
                write_channel_intel(fifo_C_drain_C_drain_L1_out_IO_0_1, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO()
{
    int p0 = 0, p1 = 1; // module id
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
                local_C[0][0] = read_channel_intel(fifo_C_drain_PE_1_0);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L1_out_IO_0_1);
                }
                write_channel_intel(fifo_C_drain_C_drain_L1_out_IO_0_2, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO()
{
    int p0 = 1, p1 = 0; // module id
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
                local_C[0][0] = read_channel_intel(fifo_C_drain_PE_0_1);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L1_out_IO_1_0);
                }
                write_channel_intel(fifo_C_drain_C_drain_L1_out_IO_1_1, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO()
{
    int p0 = 1, p1 = 1; // module id
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
                local_C[0][0] = read_channel_intel(fifo_C_drain_PE_1_1);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L1_out_IO_1_1);
                }
                write_channel_intel(fifo_C_drain_C_drain_L1_out_IO_1_2, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L2_out_IO()
{
    int p0 = 0; // module id
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
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L2_out_IO_0);
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L2_out_IO_0);
                }
                write_channel_intel(fifo_C_drain_C_drain_L2_out_IO_1, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L2_out_IO()
{
    int p0 = 1; // module id
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
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L2_out_IO_1);
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_C_drain_L2_out_IO_1);
                }
                write_channel_intel(fifo_C_drain_C_drain_L2_out_IO_2, fifo_data);
              }
              }
            }
          }
}
/* Module Definition */

/* Module Definition */
__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L3_out_IO(int *C)
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
                fifo_data = read_channel_intel(fifo_C_drain_C_drain_L2_out_IO_2);
                C[(2 * c0 + c4) * 512 + (2 * c1 + c3)] = fifo_data;
              }
              }
            }
          }
}
/* Module Definition */

