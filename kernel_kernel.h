extern "C" {
void kernel0(int *A, int *B, int *C);
}
typedef ap_uint<128> A_t4;
typedef ap_uint<128> B_t4;
void PE(int idx, int idy, hls::stream<A_t2> &fifo_A_in, hls::stream<A_t2> &fifo_A_out, hls::stream<B_t2> &fifo_B_in, hls::stream<B_t2> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void A_IO_L2_in_intra_trans(int idx, int idy, int c0_prev, int c1_prev, int c2_prev, A_t4 local_A[1][1], hls::stream<A_t2> &fifo_A_local_out);
void A_IO_L2_in_inter_trans(int idx, int idy, int c0, int c1, int c2, A_t4 local_A[1][1], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out);
void A_IO_L2_in(int idx, int idy, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<A_t2> &fifo_A_local_out);
void A_IO_L3_in(int idx, A_t4 *A, hls::stream<A_t4> &fifo_A_local_out);
void B_IO_L2_in_intra_trans(int idx, int idy, int c0_prev, int c1_prev, int c2_prev, B_t4 local_B[1][1], hls::stream<B_t2> &fifo_B_local_out);
void B_IO_L2_in_inter_trans(int idx, int idy, int c0, int c1, int c2, B_t4 local_B[1][1], hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out);
void B_IO_L2_in(int idx, int idy, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, hls::stream<B_t2> &fifo_B_local_out);
void B_IO_L3_in(int idx, B_t4 *B, hls::stream<B_t4> &fifo_B_local_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int idz, int c0, int c1, int local_C[1][1], hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int idz, int c0_prev, int c1_prev, int local_C[1][1], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out(int idx, int idy, int idz, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L2_out(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L3_out(int idx, int *C, hls::stream<int> &fifo_C_drain_local_in);
