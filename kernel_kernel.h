extern "C" {
void kernel0(int *A, int *B, int *C);
}
void PE(int idx, int idy, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void A_L2_in_IO(int idx, hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out);
void A_L3_in_IO(int *A, hls::stream<int> &fifo_A_local_out);
void B_L2_in_IO(int idx, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out);
void B_L3_in_IO(int *B, hls::stream<int> &fifo_B_local_out);
void C_drain_L1_out_IO(int idx, int idy, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_L2_out_IO(int idx, hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_L3_out_IO(int *C, hls::stream<int> &fifo_C_drain_local_in);
