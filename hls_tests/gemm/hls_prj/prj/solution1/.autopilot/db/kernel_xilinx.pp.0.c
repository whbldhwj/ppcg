# 1 "src/kernel_xilinx.c"
# 1 "src/kernel_xilinx.c" 1
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 149 "<built-in>" 3
# 1 "<command line>" 1





# 1 "/opt/tools/xilinx/Vivado/2019.2/common/technology/autopilot/etc/autopilot_ssdm_op.h" 1
# 305 "/opt/tools/xilinx/Vivado/2019.2/common/technology/autopilot/etc/autopilot_ssdm_op.h"
    void _ssdm_op_IfRead() __attribute__ ((nothrow));
    void _ssdm_op_IfWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfNbRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfNbWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfCanRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_op_IfCanWrite() __attribute__ ((nothrow));


    void _ssdm_StreamRead() __attribute__ ((nothrow));
    void _ssdm_StreamWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamNbRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamNbWrite() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamCanRead() __attribute__ ((nothrow));
    unsigned int __attribute__ ((bitwidth(1))) _ssdm_StreamCanWrite() __attribute__ ((nothrow));




    void _ssdm_op_MemShiftRead() __attribute__ ((nothrow));

    void _ssdm_op_Wait() __attribute__ ((nothrow));
    void _ssdm_op_Poll() __attribute__ ((nothrow));

    void _ssdm_op_Return() __attribute__ ((nothrow));


    void _ssdm_op_SpecSynModule() __attribute__ ((nothrow));
    void _ssdm_op_SpecTopModule() __attribute__ ((nothrow));
    void _ssdm_op_SpecProcessDecl() __attribute__ ((nothrow));
    void _ssdm_op_SpecProcessDef() __attribute__ ((nothrow));
    void _ssdm_op_SpecPort() __attribute__ ((nothrow));
    void _ssdm_op_SpecConnection() __attribute__ ((nothrow));
    void _ssdm_op_SpecChannel() __attribute__ ((nothrow));
    void _ssdm_op_SpecSensitive() __attribute__ ((nothrow));
    void _ssdm_op_SpecModuleInst() __attribute__ ((nothrow));
    void _ssdm_op_SpecPortMap() __attribute__ ((nothrow));

    void _ssdm_op_SpecReset() __attribute__ ((nothrow));

    void _ssdm_op_SpecPlatform() __attribute__ ((nothrow));
    void _ssdm_op_SpecClockDomain() __attribute__ ((nothrow));
    void _ssdm_op_SpecPowerDomain() __attribute__ ((nothrow));

    int _ssdm_op_SpecRegionBegin() __attribute__ ((nothrow));
    int _ssdm_op_SpecRegionEnd() __attribute__ ((nothrow));

    void _ssdm_op_SpecLoopName() __attribute__ ((nothrow));

    void _ssdm_op_SpecLoopTripCount() __attribute__ ((nothrow));

    int _ssdm_op_SpecStateBegin() __attribute__ ((nothrow));
    int _ssdm_op_SpecStateEnd() __attribute__ ((nothrow));

    void _ssdm_op_SpecInterface() __attribute__ ((nothrow));

    void _ssdm_op_SpecPipeline() __attribute__ ((nothrow));
    void _ssdm_op_SpecDataflowPipeline() __attribute__ ((nothrow));


    void _ssdm_op_SpecLatency() __attribute__ ((nothrow));
    void _ssdm_op_SpecParallel() __attribute__ ((nothrow));
    void _ssdm_op_SpecProtocol() __attribute__ ((nothrow));
    void _ssdm_op_SpecOccurrence() __attribute__ ((nothrow));

    void _ssdm_op_SpecResource() __attribute__ ((nothrow));
    void _ssdm_op_SpecResourceLimit() __attribute__ ((nothrow));
    void _ssdm_op_SpecCHCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecFUCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecIFCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecIPCore() __attribute__ ((nothrow));
    void _ssdm_op_SpecKeepValue() __attribute__ ((nothrow));
    void _ssdm_op_SpecMemCore() __attribute__ ((nothrow));

    void _ssdm_op_SpecExt() __attribute__ ((nothrow));




    void _ssdm_SpecArrayDimSize() __attribute__ ((nothrow));

    void _ssdm_RegionBegin() __attribute__ ((nothrow));
    void _ssdm_RegionEnd() __attribute__ ((nothrow));

    void _ssdm_Unroll() __attribute__ ((nothrow));
    void _ssdm_UnrollRegion() __attribute__ ((nothrow));

    void _ssdm_InlineAll() __attribute__ ((nothrow));
    void _ssdm_InlineLoop() __attribute__ ((nothrow));
    void _ssdm_Inline() __attribute__ ((nothrow));
    void _ssdm_InlineSelf() __attribute__ ((nothrow));
    void _ssdm_InlineRegion() __attribute__ ((nothrow));

    void _ssdm_SpecArrayMap() __attribute__ ((nothrow));
    void _ssdm_SpecArrayPartition() __attribute__ ((nothrow));
    void _ssdm_SpecArrayReshape() __attribute__ ((nothrow));

    void _ssdm_SpecStream() __attribute__ ((nothrow));

    void _ssdm_op_SpecStable() __attribute__ ((nothrow));
    void _ssdm_op_SpecStableContent() __attribute__ ((nothrow));

    void _ssdm_op_SpecPipoDepth() __attribute__ ((nothrow));

    void _ssdm_SpecExpr() __attribute__ ((nothrow));
    void _ssdm_SpecExprBalance() __attribute__ ((nothrow));

    void _ssdm_SpecDependence() __attribute__ ((nothrow));

    void _ssdm_SpecLoopMerge() __attribute__ ((nothrow));
    void _ssdm_SpecLoopFlatten() __attribute__ ((nothrow));
    void _ssdm_SpecLoopRewind() __attribute__ ((nothrow));

    void _ssdm_SpecFuncInstantiation() __attribute__ ((nothrow));
    void _ssdm_SpecFuncBuffer() __attribute__ ((nothrow));
    void _ssdm_SpecFuncExtract() __attribute__ ((nothrow));
    void _ssdm_SpecConstant() __attribute__ ((nothrow));

    void _ssdm_DataPack() __attribute__ ((nothrow));
    void _ssdm_SpecDataPack() __attribute__ ((nothrow));

    void _ssdm_op_SpecBitsMap() __attribute__ ((nothrow));
    void _ssdm_op_SpecLicense() __attribute__ ((nothrow));
# 7 "<command line>" 2
# 1 "<built-in>" 2
# 1 "src/kernel_xilinx.c" 2
# 1 "src/kernel_kernel.h" 1
extern "C" {
void kernel0(A_t4 *A, B_t4 *B, C_t2 *C);
}
typedef ap_uint<128> A_t4;
typedef ap_uint<128> B_t4;
typedef ap_uint<64> C_t2;
void A_IO_L3_in(A_t4 *A, hls::stream<A_t4> &fifo_A_local_out);
void A_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, A_t4 local_A[2][1], hls::stream<A_t2> &fifo_A_local_out);
void A_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, A_t4 local_A[2][1], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out);
void A_IO_L2_in(int idx, hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<A_t2> &fifo_A_local_out);
void B_IO_L3_in(B_t4 *B, hls::stream<B_t4> &fifo_B_local_out);
void B_IO_L2_in_intra_trans(int idx, int c0_prev, int c1_prev, int c2_prev, B_t4 local_B[2][1], hls::stream<B_t2> &fifo_B_local_out);
void B_IO_L2_in_inter_trans(int idx, int c0, int c1, int c2, B_t4 local_B[2][1], hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out);
void B_IO_L2_in(int idx, hls::stream<B_t4> &fifo_B_in, hls::stream<B_t4> &fifo_B_out, hls::stream<B_t2> &fifo_B_local_out);
void PE(int idx, int idy, hls::stream<A_t2> &fifo_A_in, hls::stream<A_t2> &fifo_A_out, hls::stream<B_t2> &fifo_B_in, hls::stream<B_t2> &fifo_B_out, hls::stream<int> &fifo_C_drain_out);
void C_drain_IO_L1_out_intra_trans(int idx, int idy, int c0, int c1, C_t2 local_C[2][1], hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L1_out_inter_trans(int idx, int idy, int c0_prev, int c1_prev, C_t2 local_C[2][1], hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out);
void C_drain_IO_L1_out(int idx, int idy, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<int> &fifo_C_drain_local_in);
void C_drain_IO_L2_out(int idx, hls::stream<C_t2> &fifo_C_drain_in, hls::stream<C_t2> &fifo_C_drain_out, hls::stream<C_t2> &fifo_C_drain_local_in);
void C_drain_IO_L3_out(C_t2 *C, hls::stream<C_t2> &fifo_C_drain_local_in);
# 2 "src/kernel_xilinx.c" 2

void kernel0(int *A, int *B, int *C)
{
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {

          for (int c3 = 0; c3 <= 1; c3 += 1)
            for (int c4 = 0; c4 <= 1; c4 += 1) {

              {
                if (c2 >= 1) {
                  local_C[0][0] = C[(2 * c0 + c3) * 512 + (2 * c1 + c4)];
                } else {
                  local_C[0][0] = 0;
                }
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  local_C[0][0] = (local_C[0][0] + (A[(2 * c0 + c3) * 512 + (2 * c2 + c5)] * B[(2 * c2 + c5) * 512 + (2 * c1 + c4)]));
                C[(2 * c0 + c3) * 512 + (2 * c1 + c4)] = local_C[0][0];
              }
            }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void PE(int idx, int idy, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_C_drain_out)
{
    int p0 = idx, p1 = idy;
    int local_A[1];
    int local_B[1];
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {


          {
            if (c2 == 0)
              local_C[0][0] = 0;
            for (int c5 = 0; c5 <= 1; c5 += 1) {
              local_A[0] = read_channel_intel(fifo_A_in);
              local_B[0] = read_channel_intel(fifo_B_in);
              local_C[0][0] = (local_C[0][0] + (local_A[0] * local_B[0]));
              write_channel_intel(fifo_B_out, local_B[0]);
              write_channel_intel(fifo_A_out, local_A[0]);
            }
            if (c2 == 255)
              write_channel_intel(fifo_C_drain_out, local_C[0][0]);
          }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void A_L2_in_IO(int idx, channel int fifo_A_in, channel int fifo_A_out, channel int fifo_A_local_out)
{
    int p0 = idx;
    int fifo_data;
    int local_A[1][2];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {


          for (int c3 = p0; c3 <= 1; c3 += 1) {
            for (int c5 = 0; c5 <= 1; c5 += 1)
            {
              fifo_data = read_channel_intel(fifo_A_in);
              if (c3 == p0) {
                local_A[0][c5] = fifo_data;
              } else {
                write_channel_intel(fifo_A_out, fifo_data);
              }
            }

            for (int c4 = 0; c4 <= 1; c4 += 1) {

              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_A_local_out, local_A[0][c5]);
              }
            }
          }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void A_L3_in_IO(int *A, channel int fifo_A_local_out)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {


          for (int c3 = 0; c3 <= 1; c3 += 1) {

            for (int c5 = 0; c5 <= 1; c5 += 1)
            {
              fifo_data = A[(2 * c0 + c3) * 512 + (2 * c2 + c5)];
              write_channel_intel(fifo_A_local_out, fifo_data);
            }
          }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void B_L2_in_IO(int idx, channel int fifo_B_in, channel int fifo_B_out, channel int fifo_B_local_out)
{
    int p0 = idx;
    int fifo_data;
    int local_B[2][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {


          for (int c3 = p0; c3 <= 1; c3 += 1) {
            for (int c4 = 0; c4 <= 1; c4 += 1)
            {
              fifo_data = read_channel_intel(fifo_B_in);
              if (c3 == p0) {
                local_B[c4][0] = fifo_data;
              } else {
                write_channel_intel(fifo_B_out, fifo_data);
              }
            }

            for (int c4 = 0; c4 <= 1; c4 += 1) {

              {
                for (int c5 = 0; c5 <= 1; c5 += 1)
                  write_channel_intel(fifo_B_local_out, local_B[c5][0]);
              }
            }
          }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void B_L3_in_IO(int *B, channel int fifo_B_local_out)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1) {


          for (int c3 = 0; c3 <= 1; c3 += 1) {

            for (int c4 = 0; c4 <= 1; c4 += 1)
            {
              fifo_data = B[(2 * c2 + c4) * 512 + (2 * c1 + c3)];
              write_channel_intel(fifo_B_local_out, fifo_data);
            }
          }
        }
}



__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L1_out_IO(int idx, int idy, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in)
{
    int p0 = idx, p1 = idy;
    int local_C[1][1];

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {



            for (int c4 = 0; c4 <= p1; c4 += 1) {

              {
                local_C[0][0] = read_channel_intel(fifo_C_drain_local_in);
              {
                if (c4 == p1) {
                  fifo_data = local_C[0][0];
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                }
                write_channel_intel(fifo_C_drain_out, fifo_data);
              }
              }
            }
          }
}



__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L2_out_IO(int idx, channel int fifo_C_drain_in, channel int fifo_C_drain_out, channel int fifo_C_drain_local_in)
{
    int p0 = idx;
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {


            for (int c3 = 0; c3 <= p0; c3 += 1) {

              for (int c4 = 0; c4 <= 1; c4 += 1) {

              {
                if (c3 == p0) {
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                } else {
                  fifo_data = read_channel_intel(fifo_C_drain_in);
                }
                write_channel_intel(fifo_C_drain_out, fifo_data);
              }
              }
            }
          }
}



__attribute__((max_global_work_dim(0)))
__kernel void C_drain_L3_out_IO(int *C, channel int fifo_C_drain_local_in)
{
    int fifo_data;

    for (int c0 = 0; c0 <= 255; c0 += 1)
      for (int c1 = 0; c1 <= 255; c1 += 1)
        for (int c2 = 0; c2 <= 255; c2 += 1)
          if (c2 == 255) {


            for (int c3 = 0; c3 <= 1; c3 += 1) {

              for (int c4 = 0; c4 <= 1; c4 += 1) {

              {
                fifo_data = read_channel_intel(fifo_C_drain_local_in);
                C[(2 * c0 + c4) * 512 + (2 * c1 + c3)] = fifo_data;
              }
              }
            }
          }
}


void kernel0(int *A, int *B, int *C)
{

                  channel int fifo_A_PE_0_0;
                  channel int fifo_A_PE_0_1;
                  channel int fifo_A_PE_0_2;
                  channel int fifo_A_PE_1_0;
                  channel int fifo_A_PE_1_1;
                  channel int fifo_A_PE_1_2;
                  channel int fifo_B_PE_0_0;
                  channel int fifo_B_PE_1_0;
                  channel int fifo_B_PE_2_0;
                  channel int fifo_B_PE_0_1;
                  channel int fifo_B_PE_1_1;
                  channel int fifo_B_PE_2_1;
                  channel int fifo_C_drain_PE_0_0;
                  channel int fifo_C_drain_PE_1_0;
                  channel int fifo_C_drain_PE_0_1;
                  channel int fifo_C_drain_PE_1_1;
                          channel int fifo_A_A_L2_in_IO_0;
                          channel int fifo_A_A_L2_in_IO_1;
                          channel int fifo_A_A_L2_in_IO_2;
                          channel int fifo_B_B_L2_in_IO_0;
                          channel int fifo_B_B_L2_in_IO_1;
                          channel int fifo_B_B_L2_in_IO_2;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_0_0;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_0_1;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_0_2;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_1_0;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_1_1;
                                 channel int fifo_C_drain_C_drain_L1_out_IO_1_2;
                                 channel int fifo_C_drain_C_drain_L2_out_IO_0;
                                 channel int fifo_C_drain_C_drain_L2_out_IO_1;
                                 channel int fifo_C_drain_C_drain_L2_out_IO_2;



    PE(
                        0,
                        0,
                   fifo_A_PE_0_0,
                   fifo_A_PE_0_1,
                   fifo_B_PE_0_0,
                   fifo_B_PE_1_0,
                   fifo_C_drain_PE_0_0
    );



    PE(
                        0,
                        1,
                   fifo_A_PE_0_1,
                   fifo_A_PE_0_2,
                   fifo_B_PE_0_1,
                   fifo_B_PE_1_1,
                   fifo_C_drain_PE_0_1
    );



    PE(
                        1,
                        0,
                   fifo_A_PE_1_0,
                   fifo_A_PE_1_1,
                   fifo_B_PE_1_0,
                   fifo_B_PE_2_0,
                   fifo_C_drain_PE_1_0
    );



    PE(
                        1,
                        1,
                   fifo_A_PE_1_1,
                   fifo_A_PE_1_2,
                   fifo_B_PE_1_1,
                   fifo_B_PE_2_1,
                   fifo_C_drain_PE_1_1
    );



    A_L2_in_IO(
                        0,
                   fifo_A_A_L2_in_IO_0,
                   fifo_A_A_L2_in_IO_1,
                   fifo_A_PE_0_0
    );



    A_L2_in_IO(
                        1,
                   fifo_A_A_L2_in_IO_1,
                   fifo_A_A_L2_in_IO_2,
                   fifo_A_PE_1_0
    );



    A_L3_in_IO(
                    A,
                   fifo_A_A_L2_in_IO_0
    );



    B_L2_in_IO(
                        0,
                   fifo_B_B_L2_in_IO_0,
                   fifo_B_B_L2_in_IO_1,
                   fifo_B_PE_0_0
    );



    B_L2_in_IO(
                        1,
                   fifo_B_B_L2_in_IO_1,
                   fifo_B_B_L2_in_IO_2,
                   fifo_B_PE_0_1
    );



    B_L3_in_IO(
                    B,
                   fifo_B_B_L2_in_IO_0
    );



    C_drain_L1_out_IO(
                        0,
                        0,
                   fifo_C_drain_C_drain_L1_out_IO_0_0,
                   fifo_C_drain_C_drain_L1_out_IO_0_1,
                   fifo_C_drain_PE_0_0
    );



    C_drain_L1_out_IO(
                        0,
                        1,
                   fifo_C_drain_C_drain_L1_out_IO_0_1,
                   fifo_C_drain_C_drain_L1_out_IO_0_2,
                   fifo_C_drain_PE_1_0
    );



    C_drain_L1_out_IO(
                        1,
                        0,
                   fifo_C_drain_C_drain_L1_out_IO_1_0,
                   fifo_C_drain_C_drain_L1_out_IO_1_1,
                   fifo_C_drain_PE_0_1
    );



    C_drain_L1_out_IO(
                        1,
                        1,
                   fifo_C_drain_C_drain_L1_out_IO_1_1,
                   fifo_C_drain_C_drain_L1_out_IO_1_2,
                   fifo_C_drain_PE_1_1
    );



    C_drain_L2_out_IO(
                        0,
                   fifo_C_drain_C_drain_L2_out_IO_0,
                   fifo_C_drain_C_drain_L2_out_IO_1,
                   fifo_C_drain_C_drain_L1_out_IO_0_2
    );



    C_drain_L2_out_IO(
                        1,
                   fifo_C_drain_C_drain_L2_out_IO_1,
                   fifo_C_drain_C_drain_L2_out_IO_2,
                   fifo_C_drain_C_drain_L1_out_IO_1_2
    );



    C_drain_L3_out_IO(
                    C,
                   fifo_C_drain_C_drain_L2_out_IO_2
    );


}
