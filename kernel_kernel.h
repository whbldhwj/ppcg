void kernel0(int *A, int *B, int *C);
void PE(hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_drain_out, int idx, int idy);
void A_L2_in_IO(hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_A_local_out, int idx);
void A_L3_in_IO(int *A, hls::stream<int> &fifo_A_local_out);
void B_L2_in_IO(hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_B_local_out, int idx);
void B_L3_in_IO(int *B, hls::stream<int> &fifo_B_local_out);
void C_drain_L1_out_IO(hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in, int idx, int idy);
void C_drain_L2_out_IO(hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in, int idx);
void C_drain_L3_out_IO(int *C, hls::stream<int> &fifo_C_drain_local_in);
