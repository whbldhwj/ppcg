void kernel0(int *A, int *B, int *C);
void C_drain_L1_out_IO(hls::stream<int> &fifo_C_drain_in, hls::stream<int> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in, int idx, int idy);
