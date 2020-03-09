extern "C" {
void kernel0(int *A, int *B, int *C);
}
void PE(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void A_IO_L2_in_intra_trans(int idx, int idy, int c0_prev, int c1_prev, int c2_prev, int local_A[1][4], hls::stream<int> &fifo_A_local_out);
void A_IO_L2_in_inter_trans(int idx, int idy, int c0, int c1, int c2, int local_A[1][4], hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out);
void A_IO_L2_in(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out);
void A_IO_L3_in(int idx, int *A, hls::stream<int> &fifo_A_local_out);
void B_IO_L2_in_intra_trans(int idx, int idy, int c0_prev, int c1_prev, int c2_prev, int local_B[4][1], hls::stream<int> &fifo_B_local_out);
void B_IO_L2_in_inter_trans(int idx, int idy, int c0, int c1, int c2, int local_B[4][1], hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out);
void B_IO_L2_in(int idx, int idy, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out);
void B_IO_L3_in(int idx, int *B, hls::stream<int> &fifo_B_local_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int idz, int c0, int c1, int local_C[1][1], hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int idz, int c0_prev, int c1_prev, int local_C[1][1], hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out(int idx, int idy, int idz, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L2_out(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L3_out(int idx, int *C, hls::stream<int> &fifo_C_drain_local_in);
