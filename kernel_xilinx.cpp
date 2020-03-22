#include "kernel_kernel.h"
/* Module Definition */
void A_IO_L3_in(A_t4 *A, hls::stream<A_t4> &fifo_A_local_out)
{

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // io_L3
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L2
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              #pragma HLS PIPELINE II=1
            {
              A_t4 fifo_data;
              fifo_data = A[((4 * c0 + 2 * c3 + c4) * 8 + 4 * c2) / 4];
              fifo_A_local_out.write(fifo_data);
            }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, A_t4 local_A[2][1], hls::stream<A_t2> &fifo_A_local_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    // io_L2
    // io_L1
    // pe
    for (int c5 = 0; c5 <= 1; c5 += 1) {
      // latency
      for (int c6 = 0; c6 <= 1; c6 += 1) {
        // latency
        for (int c7 = 0; c7 <= 1; c7 += 1) {
          // simd
          #pragma HLS PIPELINE II=1
          {
            A_t2 fifo_data;
            A_t4 buf_data;
            A_t2 buf_data_split[2];
            #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
            buf_data = local_A[c7][2 * c5 / 4];
            for (int n = 0; n < 2; n++) {
                #pragma HLS UNROLL
                buf_data_split[n] = buf_data(63, 0);
                buf_data = buf_data >> 64;
            }
            int split_i = (2 * c5 / 2) % 2;
            fifo_data = buf_data_split[split_i];
            fifo_A_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void A_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    for (int c3 = p0; c3 <= 1; c3 += 1) {
      // io_L2
      for (int c4 = 0; c4 <= 1; c4 += 1) {
        #pragma HLS PIPELINE II=1
        {
          A_t4 fifo_data;
          fifo_data = fifo_A_in.read();
          if (c3 == p0) {
            local_A[c4][0 / 4] = fifo_data;
          } else {
            fifo_A_out.write(fifo_data);
          }
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
void B_IO_L3_in(B_t4 *B, hls::stream<B_t4> &fifo_B_local_out)
{

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1)
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // io_L3
          for (int c3 = 0; c3 <= 1; c3 += 1) {
            // io_L2
            for (int c4 = 0; c4 <= 1; c4 += 1) {
              #pragma HLS PIPELINE II=1
            {
              B_t4 fifo_data;
              fifo_data = B[((4 * c1 + 2 * c3 + c4) * 8 + 4 * c2) / 4];
              fifo_B_local_out.write(fifo_data);
            }
            }
          }
        }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, B_t4 local_B[2][1], hls::stream<B_t2> &fifo_B_local_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    // io_L2
    // io_L1
    // pe
    for (int c5 = 0; c5 <= 1; c5 += 1) {
      // latency
      for (int c6 = 0; c6 <= 1; c6 += 1) {
        // latency
        for (int c7 = 0; c7 <= 1; c7 += 1) {
          // simd
          #pragma HLS PIPELINE II=1
          {
            B_t2 fifo_data;
            B_t4 buf_data;
            B_t2 buf_data_split[2];
            #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
            buf_data = local_B[c6][2 * c5 / 4];
            for (int n = 0; n < 2; n++) {
                #pragma HLS UNROLL
                buf_data_split[n] = buf_data(63, 0);
                buf_data = buf_data >> 64;
            }
            int split_i = (2 * c5 / 2) % 2;
            fifo_data = buf_data_split[split_i];
            fifo_B_local_out.write(fifo_data);
          }
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out)
{
#pragma HLS INLINE OFF
    int p0 = idx; // module id

    for (int c3 = p0; c3 <= 1; c3 += 1) {
      // io_L2
      for (int c4 = 0; c4 <= 1; c4 += 1) {
        #pragma HLS PIPELINE II=1
        {
          B_t4 fifo_data;
          fifo_data = fifo_B_in.read();
          if (c3 == p0) {
            local_B[c4][0 / 4] = fifo_data;
          } else {
            fifo_B_out.write(fifo_data);
          }
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
void PE(int idx, int idy, hls::stream<A_t2> &fifo_A_in, hls::stream<A_t2> &fifo_A_out, hls::stream<B_t2> &fifo_B_in, hls::stream<B_t2> &fifo_B_out, hls::stream<int> &fifo_C_drain_out)
{
    int p0 = idx, p1 = idy; // module id
    int local_A[1][2];
    #pragma HLS ARRAY_PARTITION variable=local_A dim=2 factor=2 cyclic
    int local_B[1][2];
    #pragma HLS ARRAY_PARTITION variable=local_B dim=2 factor=2 cyclic
    int local_C[2][2];

    for (int c0 = 0; c0 <= 1; c0 += 1)
      for (int c1 = 0; c1 <= 1; c1 += 1) {
        // array
        // pe
        // latency
        for (int c6 = 0; c6 <= 1; c6 += 1) {
          // latency
          for (int c7 = 0; c7 <= 1; c7 += 1) {
          #pragma HLS PIPELINE II=1
            // simd
            // hls_unroll
            local_C[c7][c6] = 0;
          }
        }
        for (int c2 = 0; c2 <= 1; c2 += 1) {
          // array
          // pe
          for (int c5 = 0; c5 <= 1; c5 += 1) {
            // latency
            for (int c6 = 0; c6 <= 1; c6 += 1) {
              // latency
              for (int c7 = 0; c7 <= 1; c7 += 1) {
              #pragma HLS PIPELINE II=1
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
                for (int c8 = 0; c8 <= 1; c8 += 1)
                #pragma HLS UNROLL
                  local_C[c7][c6] = (local_C[c7][c6] + (local_A[0][c8] * local_B[0][c8]));
                if (c2 == 1 && c5 == 1)
                  fifo_C_drain_out.write(local_C[c7][c6]);
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
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, C_t2 local_C[2][1], hls::stream<int> &fifo_C_drain_local_in)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    // io_L1
    // pe
    // latency
    for (int c6 = 0; c6 <= 1; c6 += 1) {
      // latency
      for (int c7 = 0; c7 <= 1; c7 += 1) {
        // simd
        #pragma HLS PIPELINE II=1
        {
          int fifo_data;
          C_t2 buf_data;
          ap_uint<32> buf_data_split[2];
          #pragma HLS ARRAY_PARTITION variable=buf_data_split complete
          buf_data = local_C[c7][c6 / 2];
          for (int n = 0; n < 2; n++) {
              #pragma HLS UNROLL
              buf_data_split[n] = buf_data(31, 0);
              buf_data = buf_data >> 32;
          }
          int split_i = (c6 / 1) % 1;
          fifo_data = fifo_C_drain_local_in.read();
          buf_data_split[split_i] = Reinterpret<ap_uint<32> >(fifo_data);
          buf_data = (buf_data_split[0], buf_data_split[1]);
          local_C[c7][c6 / 2] = buf_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0_prev, int c1_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out)
{
#pragma HLS INLINE OFF
    int p0 = idx, p1 = idy; // module id

    for (int c4 = 0; c4 <= p1; c4 += 1) {
      // io_L1
      for (int c5 = 0; c5 <= 1; c5 += 1) {
        #pragma HLS PIPELINE II=1
        {
          C_t2 fifo_data;
          if (c4 == p1) {
            fifo_data = local_C[c5][0 / 2];
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
void C_drain_IO_L1_out(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy; // module id
    C_t2 local_C_ping[2][1];
    C_t2 local_C_pong[2][1];
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
void C_drain_IO_L2_out(int idx, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in)
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
              #pragma HLS PIPELINE II=1
              {
                C_t2 fifo_data;
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
void C_drain_IO_L3_out(C_t2 *C, hls::stream<C_t2> &fifo_C_drain_local_in)
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
              #pragma HLS PIPELINE II=1
            {
              C_t2 fifo_data;
              fifo_data = fifo_C_drain_local_in.read();
              C[((4 * c0 + 2 * c4 + c5) * 8 + (4 * c1 + 2 * c3)) / 2] = fifo_data;
            }
            }
          }
        }
      }
}
/* Module Definition */

extern "C" {
void kernel0(A_t4 *A, B_t4 *B, C_t2 *C)
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
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_0 depth=1
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_1 depth=1
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_2 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_0 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_1 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_2 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_0_0;
    #pragma HLS STREAM variable=fifo_A_PE_0_0 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_0_1;
    #pragma HLS STREAM variable=fifo_A_PE_0_1 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_0_2;
    #pragma HLS STREAM variable=fifo_A_PE_0_2 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_0;
    #pragma HLS STREAM variable=fifo_A_PE_1_0 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_1;
    #pragma HLS STREAM variable=fifo_A_PE_1_1 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_2;
    #pragma HLS STREAM variable=fifo_A_PE_1_2 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_0_0;
    #pragma HLS STREAM variable=fifo_B_PE_0_0 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_1_0;
    #pragma HLS STREAM variable=fifo_B_PE_1_0 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_2_0;
    #pragma HLS STREAM variable=fifo_B_PE_2_0 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_0_1;
    #pragma HLS STREAM variable=fifo_B_PE_0_1 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_1_1;
    #pragma HLS STREAM variable=fifo_B_PE_1_1 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_2_1;
    #pragma HLS STREAM variable=fifo_B_PE_2_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_2 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_2 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_0 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_1 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_2;
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
}
