#include <ap_int.h>
#include <hls_stream.h>

template<typename To, typename From>
inline To Reinterpret(const From &val) {
  return reinterpret_cast<const To &>(val);
}

typedef ap_uint<64> A_t2;
typedef ap_uint<128> A_t4;
typedef ap_uint<64> B_t2;
typedef ap_uint<128> B_t4;
typedef ap_uint<64> C_t2;
extern "C" {
void kernel0(A_t4 *A, B_t4 *B, C_t2 *C);
}
void A_IO_L3_in(A_t4 *A, hls::stream<A_t4> &fifo_A_local_out);
void A_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, A_t4 local_A[2][1], hls::stream<A_t2> &fifo_A_local_out, bool intra_trans_en);
void A_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, bool inter_trans_en);
void A_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, bool inter_trans_en);
void A_IO_L2_in(int idx, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<A_t2> &fifo_A_local_out);
void A_IO_L2_in_boundary(int idx, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t2> &fifo_A_local_out);
void B_IO_L3_in(B_t4 *B, hls::stream<B_t4> &fifo_B_local_out);
void B_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, B_t4 local_B[2][1], hls::stream<B_t2> &fifo_B_local_out, bool intra_trans_en);
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, bool inter_trans_en);
void B_IO_L2_in_inter_trans_boundary(int idx, int c0, int c1, int c2, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, bool inter_trans_en);
void B_IO_L2_in(int idx, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, hls::stream<B_t2> &fifo_B_local_out);
void B_IO_L2_in_boundary(int idx, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t2> &fifo_B_local_out);
void PE(int idx, int idy, hls::stream<A_t2> &fifo_A_in, hls::stream<A_t2> &fifo_A_out, hls::stream<B_t2> &fifo_B_in, hls::stream<B_t2> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, C_t2 local_C[2][1], hls::stream<int> &fifo_C_drain_local_in, bool intra_trans_en);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0_prev, int c1_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, bool inter_trans_en);
void C_drain_IO_L1_out_inter_trans_boundary(int idx, int idy, int c0_prev, int c1_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_out, bool inter_trans_en);
void C_drain_IO_L1_out(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_boundary(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L2_out(int idx, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in);
void C_drain_IO_L2_out_boundary(int idx, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in);
void C_drain_IO_L3_out(C_t2 *C, hls::stream<C_t2> &fifo_C_drain_local_in);
