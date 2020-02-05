__global__ void kernel0(int *A, int *B, int *C);
void PE(hls::stream<int> &fifo_A_in, hls::stream<int> &fifo_A_out, hls::stream<int> &fifo_B_in, hls::stream<int> &fifo_B_out, hls::stream<int> &fifo_C_in, hls::stream<int> &fifo_C_out, hls::stream<int> &fifo_C_drain_out, int idx, int idy);
