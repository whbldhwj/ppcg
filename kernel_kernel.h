__kernel void PE(int idx, int idy, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_C_drain_out);
__kernel void A_L2_in_IO(int idx, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_A_local_out);
__kernel void A_L3_in_IO(global int *A, channel int fifo_A_local_out);
__kernel void B_L2_in_IO(int idx, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_B_local_out);
__kernel void B_L3_in_IO(global int *B, channel int fifo_B_local_out);
__kernel void C_drain_L1_out_IO(int idx, int idy, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in);
__kernel void C_drain_L2_out_IO(int idx, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in);
__kernel void C_drain_L3_out_IO(global int *C, channel int fifo_C_drain_local_in);
