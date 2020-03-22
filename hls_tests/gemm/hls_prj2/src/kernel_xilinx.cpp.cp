#include "kernel_kernel.h"
/* Module Definition */
void A_IO_L3_in(int *A, hls::stream<int> &fifo_A_local_out)
{

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // io_L3
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L2
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              for (int c5 = 0; c5 <= 3; c5 += 1)
              #pragma HLS PIPELINE II=1
              {
                int fifo_data;
                fifo_data = A[(4 * c0 + 2 * c3 + c4) * 8 + (4 * c2 + c5)];
                fifo_A_local_out.write(fifo_data);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, int local_A[2][4], hls::stream<int> &fifo_A_local_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    // io_L2
    // io_L1
    // pe
    for (int c5 = 0; c5 <= 3; c5 += 1) {
      // latency
      for (int c6 = 0; c6 <= 1; c6 += 1) {
        // latency
        for (int c7 = 0; c7 <= 1; c7 += 1) {
          #pragma HLS PIPELINE II=1
          {
            int fifo_data;
            fifo_data = local_A[c7][c5];
            fifo_A_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, int local_A[2][4], hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    for (int c3 = p0; c3 <= 1; c3 += 1) {
      // io_L2
      for (int c4 = 0; c4 <= 1; c4 += 1) {
        for (int c5 = 0; c5 <= 3; c5 += 1)
        #pragma HLS PIPELINE II=1
          {
            int fifo_data;
            fifo_data = fifo_A_in.read();
            if (c3 == p0) {
              local_A[c4][c5] = fifo_data;
            } else {
              fifo_A_out.write(fifo_data);
            }
          }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in(int idx, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out)
{
    int p0 = idx; // module id
    int local_A_ping[2][4];
    int local_A_pong[2][4];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c0, c0_prev;
    int c1, c1_prev;
    int c2, c2_prev;

    {
      for (int c0 = 0; c0 <= 1; c0 += 1)
        for (int c1 = 0; c1 <= 1; c1 += 1)
          for (int c2 = 0; c2 <= 1; c2 += 1) {
            // array
            // io_L3
            {
              if (inter_trans_en)
                  A_IO_L2_in_inter_trans(idx, c0, c1, c2, arb == 0? local_A_pong : local_A_ping, fifo_A_in, fifo_A_out);
              if (intra_trans_en)
                  A_IO_L2_in_intra_trans(idx, c0_prev, c1_prev, c2_prev, arb == 0? local_A_ping : local_A_pong, fifo_A_local_out);
              intra_trans_en = 1;
              arb = !arb;
              c0_prev = c0;
              c1_prev = c1;
              c2_prev = c2;
            }
          }
      if (intra_trans_en)
          A_IO_L2_in_intra_trans(idx, c0_prev, c1_prev, c2_prev, arb == 0? local_A_ping : local_A_pong, fifo_A_local_out);
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L3_in(int *B, hls::stream<int> &fifo_B_local_out)
{

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // io_L3
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L2
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              for (int c5 = 0; c5 <= 3; c5 += 1)
              #pragma HLS PIPELINE II=1
              {
                int fifo_data;
                fifo_data = B[(4 * c1 + 2 * c3 + c4) * 8 + (4 * c2 + c5)];
                fifo_B_local_out.write(fifo_data);
              }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, int local_B[2][4], hls::stream<int> &fifo_B_local_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    // io_L2
    // io_L1
    // pe
    for (int c5 = 0; c5 <= 3; c5 += 1) {
      // latency
      for (int c6 = 0; c6 <= 1; c6 += 1) {
        // latency
        for (int c7 = 0; c7 <= 1; c7 += 1) {
          #pragma HLS PIPELINE II=1
          {
            int fifo_data;
            fifo_data = local_B[c6][c5];
            fifo_B_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, int local_B[2][4], hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    for (int c3 = p0; c3 <= 1; c3 += 1) {
      // io_L2
      for (int c4 = 0; c4 <= 1; c4 += 1) {
        for (int c5 = 0; c5 <= 3; c5 += 1)
        #pragma HLS PIPELINE II=1
          {
            int fifo_data;
            fifo_data = fifo_B_in.read();
            if (c3 == p0) {
              local_B[c4][c5] = fifo_data;
            } else {
              fifo_B_out.write(fifo_data);
            }
          }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in(int idx, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out)
{
    int p0 = idx; // module id
    int local_B_ping[2][4];
    int local_B_pong[2][4];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c0, c0_prev;
    int c1, c1_prev;
    int c2, c2_prev;

    {
      for (int c0 = 0; c0 <= 1; c0 += 1)
        for (int c1 = 0; c1 <= 1; c1 += 1)
          for (int c2 = 0; c2 <= 1; c2 += 1) {
            // array
            // io_L3
            {
              if (inter_trans_en)
                  B_IO_L2_in_inter_trans(idx, c0, c1, c2, arb == 0? local_B_pong : local_B_ping, fifo_B_in, fifo_B_out);
              if (intra_trans_en)
                  B_IO_L2_in_intra_trans(idx, c0_prev, c1_prev, c2_prev, arb == 0? local_B_ping : local_B_pong, fifo_B_local_out);
              intra_trans_en = 1;
              arb = !arb;
              c0_prev = c0;
              c1_prev = c1;
              c2_prev = c2;
            }
          }
      if (intra_trans_en)
          B_IO_L2_in_intra_trans(idx, c0_prev, c1_prev, c2_prev, arb == 0? local_B_ping : local_B_pong, fifo_B_local_out);
    }
}
/* Module Definition */

/* Module Definition */
void PE(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
{
    int p0 = idx, p1 = idy; // module id
    int local_A[1][1];
    int local_B[1][1];
    int local_C[2][2];

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // array
        // pe
        // latency
        for (int c6 = 0; c6 <= 1; c6 += 1) {
          // latency
          for (int c7 = 0; c7 <= 1; c7 += 1)
          #pragma HLS PIPELINE II=1
            local_C[c7][c6] = 0;
        }
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // pe
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // latency
            for (int c6 = 0; c6 <= 1; c6 += 1) {
              // latency
              for (int c7 = 0; c7 <= 1; c7 += 1) {
              #pragma HLS PIPELINE II=1
                local_A[0][0] = fifo_A_in.read();
                local_B[0][0] = fifo_B_in.read();
                local_C[c7][c6] = (local_C[c7][c6] + (local_A[0][0] * local_B[0][0]));
                if (c2 == 1 && c5 == 3)
                  fifo_C_drain_out.write(local_C[c7][c6]);
                fifo_B_out.write(local_B[0][0]);
                fifo_A_out.write(local_A[0][0]);
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, int local_C[2][2], hls::stream<int> &fifo_C_drain_local_in)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    // io_L1
    // pe
    // latency
    for (int c6 = 0; c6 <= 1; c6 += 1) {
      // latency
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        #pragma HLS PIPELINE II=1
        {
          int fifo_data;
          fifo_data = fifo_C_drain_local_in.read();
          local_C[c7][c6] = fifo_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0_prev, int c1_prev, int local_C[2][2], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    for (int c4 = 0; c4 <= p1; c4 += 1) {
      // io_L1
      for (int c5 = 0; c5 <= 1; c5 += 1) {
        for (int c6 = 0; c6 <= 1; c6 += 1)
        #pragma HLS PIPELINE II=1
          {
            int fifo_data;
            if (c4 == p1) {
              fifo_data = local_C[c5][c6];
            } else {
              fifo_data = fifo_C_drain_in.read();
            }
            fifo_C_drain_out.write(fifo_data);
          }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id
    int local_C_ping[2][2];
    int local_C_pong[2][2];
    bool arb = 0;
    bool inter_trans_en = 0;
    bool intra_trans_en = 1;
    int c0, c0_prev;
    int c1, c1_prev;

    {
      for (int c0 = 0; c0 <= 1; c0 += 1)
        for (int c1 = 0; c1 <= 1; c1 += 1) {
          // array
          // io_L3
          // io_L2
          {
            if (intra_trans_en)
                C_drain_IO_L1_out_intra_trans(idx, idy, c0, c1, arb == 0? local_C_ping : local_C_pong, fifo_C_drain_local_in);
            if (inter_trans_en)
                C_drain_IO_L1_out_inter_trans(idx, idy, c0_prev, c1_prev, arb == 0? local_C_pong : local_C_ping, fifo_C_drain_in, fifo_C_drain_out);
            inter_trans_en = 1;
            arb = !arb;
            c0_prev = c0;
            c1_prev = c1;
          }
        }
      if (inter_trans_en)
          C_drain_IO_L1_out_inter_trans(idx, idy, c0_prev, c1_prev, arb == 0? local_C_pong : local_C_ping, fifo_C_drain_in, fifo_C_drain_out);
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L2_out(int idx, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx; // module id

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // array
        // io_L3
        for (int c3 = 0; c3 <= p0; c3 += 1) {
          // io_L2
          for (int c4 = 0; c4 <= 1; c4 += 1) {
            // io_L1
            for (int c5 = 0; c5 <= 1; c5 += 1) {
              for (int c6 = 0; c6 <= 1; c6 += 1)
              #pragma HLS PIPELINE II=1
                {
                  int fifo_data;
                  if (c3 == p0) {
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
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L3_out(int *C, hls::stream<int> &fifo_C_drain_local_in)
{

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // array
        // io_L3
        for (int c3 = 0; c3 <= 1; c3 += 1) {
          // io_L2
          for (int c4 = 0; c4 <= 1; c4 += 1) {
            // io_L1
            for (int c5 = 0; c5 <= 1; c5 += 1) {
              for (int c6 = 0; c6 <= 1; c6 += 1)
              #pragma HLS PIPELINE II=1
              {
                int fifo_data;
                fifo_data = fifo_C_drain_local_in.read();
                C[(4 * c0 + 2 * c4 + c5) * 8 + (4 * c1 + 2 * c3 + c6)] = fifo_data;
              }
            }
          }
        }
      }
}
/* Module Definition */

//extern "C" {
void kernel0(int *A, int *B, int *C)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    /* FIFO Declaration */
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_0 depth=1
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_1 depth=1
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_2 depth=1
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_0 depth=1
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_1 depth=1
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_2 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_0;
    #pragma HLS STREAM variable=fifo_A_PE_0_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_1;
    #pragma HLS STREAM variable=fifo_A_PE_0_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_2;
    #pragma HLS STREAM variable=fifo_A_PE_0_2 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_0;
    #pragma HLS STREAM variable=fifo_A_PE_1_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_1;
    #pragma HLS STREAM variable=fifo_A_PE_1_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_2;
    #pragma HLS STREAM variable=fifo_A_PE_1_2 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_0;
    #pragma HLS STREAM variable=fifo_B_PE_0_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_0;
    #pragma HLS STREAM variable=fifo_B_PE_1_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_0;
    #pragma HLS STREAM variable=fifo_B_PE_2_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_1;
    #pragma HLS STREAM variable=fifo_B_PE_0_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_1;
    #pragma HLS STREAM variable=fifo_B_PE_1_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_1;
    #pragma HLS STREAM variable=fifo_B_PE_2_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_2 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_2 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_0 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_1 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_2 depth=1
    /* FIFO Declaration */

    /* Module Call */
    A_IO_L3_in(
        /* array */ A,
        /* fifo */ fifo_A_A_IO_L2_in_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 0,
        /* fifo */ fifo_A_A_IO_L2_in_0,
        /* fifo */ fifo_A_A_IO_L2_in_1,
        /* fifo */ fifo_A_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 1,
        /* fifo */ fifo_A_A_IO_L2_in_1,
        /* fifo */ fifo_A_A_IO_L2_in_2,
        /* fifo */ fifo_A_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L3_in(
        /* array */ B,
        /* fifo */ fifo_B_B_IO_L2_in_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 0,
        /* fifo */ fifo_B_B_IO_L2_in_0,
        /* fifo */ fifo_B_B_IO_L2_in_1,
        /* fifo */ fifo_B_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 1,
        /* fifo */ fifo_B_B_IO_L2_in_1,
        /* fifo */ fifo_B_B_IO_L2_in_2,
        /* fifo */ fifo_B_PE_0_1
    );
    /* Module Call */

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
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_2,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_2,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L3_out(
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_2
    );
    /* Module Call */

}
//}
