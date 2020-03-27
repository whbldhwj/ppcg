#include "kernel_kernel.h"
/* Module Definition */
void A_IO_L3_in(A_t16 *A, hls::stream<A_t4> &fifo_A_local_out)
{
    A_t16 local_A[4][1];

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1) {
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // hls_pipeline
      {
        A_t16 fifo_data;
        fifo_data = A[((4 * c3 + c4) * 16 + 0) / 16];
        local_A[c4][0 / 16] = fifo_data;
      }
      }
      for (int c4 = 0; c4 <= 3; c4 += 1)
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // array
          // io_L3
          for (int c6 = 0; c6 <= 1; c6 += 1) {
            // io_L2
            for (int c7 = 0; c7 <= 1; c7 += 1) {
              // hls_pipeline
              {
                A_t4 fifo_data;
                A_t16 buf_data;
                A_t4 buf_data_split[4];
                #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
                buf_data = local_A[2 * c6 + c7][4 * c5 / 16];
                for (int n = 0; n < 4; n++) {
                    #pragma HLS UNROLL
                    buf_data_split[n] = buf_data(127, 0);
                    buf_data = buf_data >> 128;
                }
                int split_i = (4 * c5 / 4) % 4;
                fifo_data = buf_data_split[split_i];
                fifo_A_local_out.write(fifo_data);
              }
            }
          }
        }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_intra_trans(int idx, int c3_prev, int c4_prev, int c5_prev, A_t4 local_A[2][1], hls::stream<A_t2> &fifo_A_local_out, bool intra_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!intra_trans_en) return;

    // io_L2
    // io_L1
    // pe
    for (int c8 = 0; c8 <= 1; c8 += 1) {
      // latency
      for (int c9 = 0; c9 <= 1; c9 += 1) {
        // latency
        for (int c10 = 0; c10 <= 1; c10 += 1) {
          // simd
          // hls_pipeline
          {
            A_t2 fifo_data;
            A_t4 buf_data;
            A_t2 buf_data_split[2];
            #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
            buf_data = local_A[c10][2 * c8 / 4];
            for (int n = 0; n < 2; n++) {
                #pragma HLS UNROLL
                buf_data_split[n] = buf_data(63, 0);
                buf_data = buf_data >> 64;
            }
            int split_i = (2 * c8 / 2) % 2;
            fifo_data = buf_data_split[split_i];
            fifo_A_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans(int idx, int c3, int c4, int c5, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!inter_trans_en) return;

    for (int c6 = p0; c6 <= 1; c6 += 1) {
      // io_L2
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        // hls_pipeline
        {
          A_t4 fifo_data;
          fifo_data = fifo_A_in.read();
          if (c6 == p0) {
            local_A[c7][0 / 4] = fifo_data;
          } else {
            fifo_A_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans_boundary(int idx, int c3, int c4, int c5, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!inter_trans_en) return;

    for (int c6 = p0; c6 <= 1; c6 += 1) {
      // io_L2
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        // hls_pipeline
        {
          A_t4 fifo_data;
          fifo_data = fifo_A_in.read();
          local_A[c7][0 / 4] = fifo_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in(int idx, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<A_t2> &fifo_A_local_out)
{
    int p0 = idx; // module id
    A_t4 local_A_ping[2][1];
    A_t4 local_A_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c3, c3_prev;
    int c4, c4_prev;
    int c5, c5_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1)
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // array
            // io_L3
            {
              if (arb == 0) {
                  A_IO_L2_in_inter_trans(idx, c3, c4, c5, local_A_pong, fifo_A_in, fifo_A_out, inter_trans_en);
                  A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_ping, fifo_A_local_out, intra_trans_en);
              } else {
                  A_IO_L2_in_inter_trans(idx, c3, c4, c5, local_A_ping, fifo_A_in, fifo_A_out, inter_trans_en);
                  A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_pong, fifo_A_local_out, intra_trans_en);
              }
              intra_trans_en = 1;
              arb = !arb;
              c3_prev = c3;
              c4_prev = c4;
              c5_prev = c5;
            }
          }
      if (arb == 0) {
          A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_ping, fifo_A_local_out, intra_trans_en);
      } else {
          A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_pong, fifo_A_local_out, intra_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_boundary(int idx, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t2> &fifo_A_local_out)
{
    int p0 = idx; // module id
    A_t4 local_A_ping[2][1];
    A_t4 local_A_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c3, c3_prev;
    int c4, c4_prev;
    int c5, c5_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1)
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // array
            // io_L3
            {
              if (arb == 0) {
                  A_IO_L2_in_inter_trans_boundary(idx, c3, c4, c5, local_A_pong, fifo_A_in, inter_trans_en);
                  A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_ping, fifo_A_local_out, intra_trans_en);
              } else {
                  A_IO_L2_in_inter_trans_boundary(idx, c3, c4, c5, local_A_ping, fifo_A_in, inter_trans_en);
                  A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_pong, fifo_A_local_out, intra_trans_en);
              }
              intra_trans_en = 1;
              arb = !arb;
              c3_prev = c3;
              c4_prev = c4;
              c5_prev = c5;
            }
          }
      if (arb == 0) {
          A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_ping, fifo_A_local_out, intra_trans_en);
      } else {
          A_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_A_pong, fifo_A_local_out, intra_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L3_in(B_t16 *B, hls::stream<B_t4> &fifo_B_local_out)
{
    B_t16 local_B[4][1];

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // hls_pipeline
        {
          B_t16 fifo_data;
          fifo_data = B[((4 * c4 + c5) * 16 + 0) / 16];
          local_B[c5][0 / 16] = fifo_data;
        }
        }
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // array
          // io_L3
          for (int c6 = 0; c6 <= 1; c6 += 1) {
            // io_L2
            for (int c7 = 0; c7 <= 1; c7 += 1) {
              // hls_pipeline
              {
                B_t4 fifo_data;
                B_t16 buf_data;
                B_t4 buf_data_split[4];
                #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
                buf_data = local_B[2 * c6 + c7][4 * c5 / 16];
                for (int n = 0; n < 4; n++) {
                    #pragma HLS UNROLL
                    buf_data_split[n] = buf_data(127, 0);
                    buf_data = buf_data >> 128;
                }
                int split_i = (4 * c5 / 4) % 4;
                fifo_data = buf_data_split[split_i];
                fifo_B_local_out.write(fifo_data);
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_intra_trans(int idx, int c3_prev, int c4_prev, int c5_prev, B_t4 local_B[2][1], hls::stream<B_t2> &fifo_B_local_out, bool intra_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!intra_trans_en) return;

    // io_L2
    // io_L1
    // pe
    for (int c8 = 0; c8 <= 1; c8 += 1) {
      // latency
      for (int c9 = 0; c9 <= 1; c9 += 1) {
        // latency
        for (int c10 = 0; c10 <= 1; c10 += 1) {
          // simd
          // hls_pipeline
          {
            B_t2 fifo_data;
            B_t4 buf_data;
            B_t2 buf_data_split[2];
            #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
            buf_data = local_B[c9][2 * c8 / 4];
            for (int n = 0; n < 2; n++) {
                #pragma HLS UNROLL
                buf_data_split[n] = buf_data(63, 0);
                buf_data = buf_data >> 64;
            }
            int split_i = (2 * c8 / 2) % 2;
            fifo_data = buf_data_split[split_i];
            fifo_B_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans(int idx, int c3, int c4, int c5, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!inter_trans_en) return;

    for (int c6 = p0; c6 <= 1; c6 += 1) {
      // io_L2
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        // hls_pipeline
        {
          B_t4 fifo_data;
          fifo_data = fifo_B_in.read();
          if (c6 == p0) {
            local_B[c7][0 / 4] = fifo_data;
          } else {
            fifo_B_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans_boundary(int idx, int c3, int c4, int c5, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    if (!inter_trans_en) return;

    for (int c6 = p0; c6 <= 1; c6 += 1) {
      // io_L2
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        // hls_pipeline
        {
          B_t4 fifo_data;
          fifo_data = fifo_B_in.read();
          local_B[c7][0 / 4] = fifo_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in(int idx, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, hls::stream<B_t2> &fifo_B_local_out)
{
    int p0 = idx; // module id
    B_t4 local_B_ping[2][1];
    B_t4 local_B_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c3, c3_prev;
    int c4, c4_prev;
    int c5, c5_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1)
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // array
            // io_L3
            {
              if (arb == 0) {
                  B_IO_L2_in_inter_trans(idx, c3, c4, c5, local_B_pong, fifo_B_in, fifo_B_out, inter_trans_en);
                  B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_ping, fifo_B_local_out, intra_trans_en);
              } else {
                  B_IO_L2_in_inter_trans(idx, c3, c4, c5, local_B_ping, fifo_B_in, fifo_B_out, inter_trans_en);
                  B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_pong, fifo_B_local_out, intra_trans_en);
              }
              intra_trans_en = 1;
              arb = !arb;
              c3_prev = c3;
              c4_prev = c4;
              c5_prev = c5;
            }
          }
      if (arb == 0) {
          B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_ping, fifo_B_local_out, intra_trans_en);
      } else {
          B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_pong, fifo_B_local_out, intra_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_boundary(int idx, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t2> &fifo_B_local_out)
{
    int p0 = idx; // module id
    B_t4 local_B_ping[2][1];
    B_t4 local_B_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 1;
    bool intra_trans_en = 0;
    int c3, c3_prev;
    int c4, c4_prev;
    int c5, c5_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1)
          for (int c5 = 0; c5 <= 3; c5 += 1) {
            // array
            // io_L3
            {
              if (arb == 0) {
                  B_IO_L2_in_inter_trans_boundary(idx, c3, c4, c5, local_B_pong, fifo_B_in, inter_trans_en);
                  B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_ping, fifo_B_local_out, intra_trans_en);
              } else {
                  B_IO_L2_in_inter_trans_boundary(idx, c3, c4, c5, local_B_ping, fifo_B_in, inter_trans_en);
                  B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_pong, fifo_B_local_out, intra_trans_en);
              }
              intra_trans_en = 1;
              arb = !arb;
              c3_prev = c3;
              c4_prev = c4;
              c5_prev = c5;
            }
          }
      if (arb == 0) {
          B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_ping, fifo_B_local_out, intra_trans_en);
      } else {
          B_IO_L2_in_intra_trans(idx, c3_prev, c4_prev, c5_prev, local_B_pong, fifo_B_local_out, intra_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void PE(int idx, int idy, hls::stream<A_t2> &fifo_A_in, hls::stream<A_t2> &fifo_A_out, hls::stream<B_t2> &fifo_B_in, hls::stream<B_t2> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
{
    int p0 = idx, p1 = idy; // module id
    int local_A[1][2];
    #pragma HLS ARRAY_PARTITION variable=local_A dim=2 factor=2 cyclic
    int local_B[1][2];
    #pragma HLS ARRAY_PARTITION variable=local_B dim=2 factor=2 cyclic
    int local_C[2][2];

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        // pe
        // latency
        for (int c9 = 0; c9 <= 1; c9 += 1) {
          // latency
          // hls_pipeline
          for (int c10 = 0; c10 <= 1; c10 += 1) {
            // simd
            // hls_unroll
            local_C[c10][c9] = 0;
          }
        }
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // array
          // pe
          for (int c8 = 0; c8 <= 1; c8 += 1) {
            // latency
            for (int c9 = 0; c9 <= 1; c9 += 1) {
              // latency
              // hls_pipeline
              for (int c10 = 0; c10 <= 1; c10 += 1) {
                {
                  A_t2 fifo_data;
                  fifo_data = fifo_A_in.read();
                  for (int n = 0; n < 2; n++) {
                  #pragma HLS UNROLL
                      local_A[0][n] = Reinterpret<int>((ap_uint<32>)fifo_data(31, 0));
                      fifo_data = fifo_data >> 32;
                  }
                }
                {
                  B_t2 fifo_data;
                  fifo_data = fifo_B_in.read();
                  for (int n = 0; n < 2; n++) {
                  #pragma HLS UNROLL
                      local_B[0][n] = Reinterpret<int>((ap_uint<32>)fifo_data(31, 0));
                      fifo_data = fifo_data >> 32;
                  }
                }
                // simd
                // hls_unroll
                for (int c11 = 0; c11 <= 1; c11 += 1)
                  local_C[c10][c9] = (local_C[c10][c9] + (local_A[0][c11] * local_B[0][c11]));
                if (c5 == 3 && c8 == 1)
                  fifo_C_drain_out.write(local_C[c10][c9]);
                {
                  B_t2 fifo_data;
                  fifo_data = (Reinterpret<ap_uint<32> >(local_B[0][1]), Reinterpret<ap_uint<32> >(local_B[0][0]));
                  fifo_B_out.write(fifo_data);
                }
                {
                  A_t2 fifo_data;
                  fifo_data = (Reinterpret<ap_uint<32> >(local_A[0][1]), Reinterpret<ap_uint<32> >(local_A[0][0]));
                  fifo_A_out.write(fifo_data);
                }
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void A_PE_dummy(int idx, int idy, hls::stream<A_t2> &fifo_A_in)
{
    int p0 = idx, p1 = idy; // module id

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        {
        }
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // array
          // pe
          for (int c8 = 0; c8 <= 1; c8 += 1) {
            // latency
            for (int c9 = 0; c9 <= 1; c9 += 1) {
              // latency
              // hls_pipeline
              for (int c10 = 0; c10 <= 1; c10 += 1) {
                A_t2 fifo_data;
                fifo_data = fifo_A_in.read();
                // simd
                // hls_unroll
                for (int c11 = 0; c11 <= 1; c11 += 1) {
                }
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void B_PE_dummy(int idx, int idy, hls::stream<B_t2> &fifo_B_in)
{
    int p0 = idx, p1 = idy; // module id

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        {
        }
        for (int c5 = 0; c5 <= 3; c5 += 1) {
          // array
          // pe
          for (int c8 = 0; c8 <= 1; c8 += 1) {
            // latency
            for (int c9 = 0; c9 <= 1; c9 += 1) {
              // latency
              // hls_pipeline
              for (int c10 = 0; c10 <= 1; c10 += 1) {
                B_t2 fifo_data;
                fifo_data = fifo_B_in.read();
                // simd
                // hls_unroll
                for (int c11 = 0; c11 <= 1; c11 += 1) {
                }
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c3, int c4, C_t2 local_C[2][1], hls::stream<int> &fifo_C_drain_local_in, bool intra_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    if (!intra_trans_en) return;

    // io_L1
    // pe
    // latency
    for (int c9 = 0; c9 <= 1; c9 += 1) {
      // latency
      for (int c10 = 0; c10 <= 1; c10 += 1) {
        // simd
        // hls_pipeline
        {
          int fifo_data;
          C_t2 buf_data;
          ap_uint<32> buf_data_split[2];
          #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
          buf_data = local_C[c10][c9 / 2];
          for (int n = 0; n < 2; n++) {
              #pragma HLS UNROLL
              buf_data_split[n] = buf_data(31, 0);
              buf_data = buf_data >> 32;
          }
          int split_i = (c9 / 1) % 2;
          fifo_data = fifo_C_drain_local_in.read();
          buf_data_split[split_i] = Reinterpret<ap_uint<32> >(fifo_data);
          buf_data = (buf_data_split[1], buf_data_split[0]);
          local_C[c10][c9 / 2] = buf_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c3_prev, int c4_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    if (!inter_trans_en) return;

    for (int c7 = p1; c7 <= 1; c7 += 1) {
      // io_L1
      for (int c8 = 0; c8 <= 1; c8 += 1) {
        // hls_dependence.local_C
        // hls_pipeline
        {
          C_t2 fifo_data;
          if (c7 == p1) {
            fifo_data = local_C[c8][0 / 2];
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
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c3_prev, int c4_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_out, bool inter_trans_en)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    if (!inter_trans_en) return;

    for (int c7 = p1; c7 <= 1; c7 += 1) {
      // io_L1
      for (int c8 = 0; c8 <= 1; c8 += 1) {
        // hls_dependence.local_C
        // hls_pipeline
        {
          C_t2 fifo_data;
          fifo_data = local_C[c8][0 / 2];
          fifo_C_drain_out.write(fifo_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id
    C_t2 local_C_ping[2][1];
    C_t2 local_C_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 0;
    bool intra_trans_en = 1;
    int c3, c3_prev;
    int c4, c4_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1) {
          // array
          // io_L3
          // io_L2
          {
            if (arb == 0) {
                C_drain_IO_L1_out_intra_trans(idx, idy, c3, c4, local_C_ping, fifo_C_drain_local_in, intra_trans_en);
                C_drain_IO_L1_out_inter_trans(idx, idy, c3_prev, c4_prev, local_C_pong, fifo_C_drain_in, fifo_C_drain_out, inter_trans_en);
            } else {
                C_drain_IO_L1_out_intra_trans(idx, idy, c3, c4, local_C_pong, fifo_C_drain_local_in, intra_trans_en);
                C_drain_IO_L1_out_inter_trans(idx, idy, c3_prev, c4_prev, local_C_ping, fifo_C_drain_in, fifo_C_drain_out, inter_trans_en);
            }
            inter_trans_en = 1;
            arb = !arb;
            c3_prev = c3;
            c4_prev = c4;
          }
        }
      if (arb == 0) {
          C_drain_IO_L1_out_inter_trans(idx, idy, c3_prev, c4_prev, local_C_pong, fifo_C_drain_in, fifo_C_drain_out, inter_trans_en);
      } else {
          C_drain_IO_L1_out_inter_trans(idx, idy, c3_prev, c4_prev, local_C_ping, fifo_C_drain_in, fifo_C_drain_out, inter_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_boundary(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id
    C_t2 local_C_ping[2][1];
    C_t2 local_C_pong[2][1];
    bool arb = 0;
    bool inter_trans_en = 0;
    bool intra_trans_en = 1;
    int c3, c3_prev;
    int c4, c4_prev;

    {
      // array_L2
      for (int c3 = 0; c3 <= 3; c3 += 1)
        for (int c4 = 0; c4 <= 3; c4 += 1) {
          // array
          // io_L3
          // io_L2
          {
            if (arb == 0) {
                C_drain_IO_L1_out_intra_trans(idx, idy, c3, c4, local_C_ping, fifo_C_drain_local_in, intra_trans_en);
                C_drain_IO_L1_out_inter_trans_boundary(idx, idy, c3_prev, c4_prev, local_C_pong, fifo_C_drain_out, inter_trans_en);
            } else {
                C_drain_IO_L1_out_intra_trans(idx, idy, c3, c4, local_C_pong, fifo_C_drain_local_in, intra_trans_en);
                C_drain_IO_L1_out_inter_trans_boundary(idx, idy, c3_prev, c4_prev, local_C_ping, fifo_C_drain_out, inter_trans_en);
            }
            inter_trans_en = 1;
            arb = !arb;
            c3_prev = c3;
            c4_prev = c4;
          }
        }
      if (arb == 0) {
          C_drain_IO_L1_out_inter_trans_boundary(idx, idy, c3_prev, c4_prev, local_C_pong, fifo_C_drain_out, inter_trans_en);
      } else {
          C_drain_IO_L1_out_inter_trans_boundary(idx, idy, c3_prev, c4_prev, local_C_ping, fifo_C_drain_out, inter_trans_en);
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L2_out(int idx, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in)
{
    int p0 = idx; // module id

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        // io_L3
        for (int c6 = p0; c6 <= 1; c6 += 1) {
          // io_L2
          for (int c7 = 0; c7 <= 1; c7 += 1) {
            // io_L1
            // pe
            for (int c8 = 0; c8 <= 1; c8 += 1) {
              // hls_pipeline
              {
                C_t2 fifo_data;
                if (c6 == p0) {
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
void C_drain_IO_L2_out_boundary(int idx, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in)
{
    int p0 = idx; // module id

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        // io_L3
        for (int c6 = p0; c6 <= 1; c6 += 1) {
          // io_L2
          for (int c7 = 0; c7 <= 1; c7 += 1) {
            // io_L1
            // pe
            for (int c8 = 0; c8 <= 1; c8 += 1) {
              // hls_pipeline
              {
                C_t2 fifo_data;
                fifo_data = fifo_C_drain_local_in.read();
                fifo_C_drain_out.write(fifo_data);
              }
            }
          }
        }
      }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L3_out(C_t4 *C, hls::stream<C_t2> &fifo_C_drain_local_in)
{
    C_t4 local_C[4][1];

    // array_L2
    for (int c3 = 0; c3 <= 3; c3 += 1)
      for (int c4 = 0; c4 <= 3; c4 += 1) {
        // array
        // io_L3
        for (int c6 = 0; c6 <= 1; c6 += 1) {
          // io_L2
          for (int c7 = 0; c7 <= 1; c7 += 1) {
            // io_L1
            for (int c8 = 0; c8 <= 1; c8 += 1) {
              // hls_dependence.local_C
              // hls_pipeline
              {
                C_t2 fifo_data;
                C_t4 buf_data;
                C_t2 buf_data_split[2];
                #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
                buf_data = local_C[2 * c7 + c8][2 * c6 / 4];
                for (int n = 0; n < 2; n++) {
                    #pragma HLS UNROLL
                    buf_data_split[n] = buf_data(63, 0);
                    buf_data = buf_data >> 64;
                }
                int split_i = (2 * c6 / 2) % 2;
                fifo_data = fifo_C_drain_local_in.read();
                buf_data_split[split_i] = fifo_data;
                buf_data = (buf_data_split[1], buf_data_split[0]);
                local_C[2 * c7 + c8][2 * c6 / 4] = buf_data;
              }
            }
          }
        }
        for (int c6 = 0; c6 <= 3; c6 += 1) {
          // hls_dependence.local_C
          // hls_pipeline
        {
          C_t4 fifo_data;
          fifo_data = local_C[c6][0 / 4];
          C[((4 * c3 + c6) * 16 + 4 * c4) / 4] = fifo_data;
        }
        }
      }
}
/* Module Definition */

