extern "C" {
void kernel0(int *A, int *B, int *C)
{
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    /* FIFO Declaration */
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_0;
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_1;
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_2;
    /* PE fifo */ hls::stream<int> fifo_A_PE_0_3;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_1;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_2;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_3;
    /* PE fifo */ hls::stream<int> fifo_A_PE_2_0;
    /* PE fifo */ hls::stream<int> fifo_A_PE_2_1;
    /* PE fifo */ hls::stream<int> fifo_A_PE_2_2;
    /* PE fifo */ hls::stream<int> fifo_A_PE_2_3;
    /* PE fifo */ hls::stream<int> fifo_A_PE_3_0;
    /* PE fifo */ hls::stream<int> fifo_A_PE_3_1;
    /* PE fifo */ hls::stream<int> fifo_A_PE_3_2;
    /* PE fifo */ hls::stream<int> fifo_A_PE_3_3;
    /* PE fifo */ hls::stream<int> fifo_A_PE_3_4;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_3_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_3_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_2;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_2;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_2;
    /* PE fifo */ hls::stream<int> fifo_B_PE_3_2;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_3;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_3;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_3;
    /* PE fifo */ hls::stream<int> fifo_B_PE_3_3;
    /* PE fifo */ hls::stream<int> fifo_B_PE_4_3;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_2;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_2;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_2;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_2;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_3;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_3;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_2_3;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_3_3;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_0_0;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_0_1;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_0_2;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_1_0;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_1_1;
    /* A_IO_L2_in fifo */ hls::stream<int> fifo_A_A_IO_L2_in_1_2;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_0_0;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_0_1;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_0_2;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_1_0;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_1_1;
    /* B_IO_L2_in fifo */ hls::stream<int> fifo_B_B_IO_L2_in_1_2;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0_0;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0_1;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0_2;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0_3;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_0_4;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1_0;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1_1;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1_2;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1_3;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_0_1_4;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0_0;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0_1;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0_2;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0_3;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_0_4;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1_0;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1_1;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1_2;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1_3;
    /* C_drain_IO_L1_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L1_out_1_1_4;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_0_0;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_0_1;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_0_2;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_1_0;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_1_1;
    /* C_drain_IO_L2_out fifo */ hls::stream<int> fifo_C_drain_C_drain_IO_L2_out_1_2;
    /* FIFO Declaration */

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
        /* module id */ 0,
        /* module id */ 2,
        /* fifo */ fifo_A_PE_0_2,
        /* fifo */ fifo_A_PE_0_3,
        /* fifo */ fifo_B_PE_0_2,
        /* fifo */ fifo_B_PE_1_2,
        /* fifo */ fifo_C_drain_PE_0_2
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 0,
        /* module id */ 3,
        /* fifo */ fifo_A_PE_0_3,
        /* fifo */ fifo_A_PE_0_4,
        /* fifo */ fifo_B_PE_0_3,
        /* fifo */ fifo_B_PE_1_3,
        /* fifo */ fifo_C_drain_PE_0_3
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
    PE(
        /* module id */ 1,
        /* module id */ 2,
        /* fifo */ fifo_A_PE_1_2,
        /* fifo */ fifo_A_PE_1_3,
        /* fifo */ fifo_B_PE_1_2,
        /* fifo */ fifo_B_PE_2_2,
        /* fifo */ fifo_C_drain_PE_1_2
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 1,
        /* module id */ 3,
        /* fifo */ fifo_A_PE_1_3,
        /* fifo */ fifo_A_PE_1_4,
        /* fifo */ fifo_B_PE_1_3,
        /* fifo */ fifo_B_PE_2_3,
        /* fifo */ fifo_C_drain_PE_1_3
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 2,
        /* module id */ 0,
        /* fifo */ fifo_A_PE_2_0,
        /* fifo */ fifo_A_PE_2_1,
        /* fifo */ fifo_B_PE_2_0,
        /* fifo */ fifo_B_PE_3_0,
        /* fifo */ fifo_C_drain_PE_2_0
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 2,
        /* module id */ 1,
        /* fifo */ fifo_A_PE_2_1,
        /* fifo */ fifo_A_PE_2_2,
        /* fifo */ fifo_B_PE_2_1,
        /* fifo */ fifo_B_PE_3_1,
        /* fifo */ fifo_C_drain_PE_2_1
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 2,
        /* module id */ 2,
        /* fifo */ fifo_A_PE_2_2,
        /* fifo */ fifo_A_PE_2_3,
        /* fifo */ fifo_B_PE_2_2,
        /* fifo */ fifo_B_PE_3_2,
        /* fifo */ fifo_C_drain_PE_2_2
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 2,
        /* module id */ 3,
        /* fifo */ fifo_A_PE_2_3,
        /* fifo */ fifo_A_PE_2_4,
        /* fifo */ fifo_B_PE_2_3,
        /* fifo */ fifo_B_PE_3_3,
        /* fifo */ fifo_C_drain_PE_2_3
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 3,
        /* module id */ 0,
        /* fifo */ fifo_A_PE_3_0,
        /* fifo */ fifo_A_PE_3_1,
        /* fifo */ fifo_B_PE_3_0,
        /* fifo */ fifo_B_PE_4_0,
        /* fifo */ fifo_C_drain_PE_3_0
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 3,
        /* module id */ 1,
        /* fifo */ fifo_A_PE_3_1,
        /* fifo */ fifo_A_PE_3_2,
        /* fifo */ fifo_B_PE_3_1,
        /* fifo */ fifo_B_PE_4_1,
        /* fifo */ fifo_C_drain_PE_3_1
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 3,
        /* module id */ 2,
        /* fifo */ fifo_A_PE_3_2,
        /* fifo */ fifo_A_PE_3_3,
        /* fifo */ fifo_B_PE_3_2,
        /* fifo */ fifo_B_PE_4_2,
        /* fifo */ fifo_C_drain_PE_3_2
    );
    /* Module Call */

    /* Module Call */
    PE(
        /* module id */ 3,
        /* module id */ 3,
        /* fifo */ fifo_A_PE_3_3,
        /* fifo */ fifo_A_PE_3_4,
        /* fifo */ fifo_B_PE_3_3,
        /* fifo */ fifo_B_PE_4_3,
        /* fifo */ fifo_C_drain_PE_3_3
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_A_A_IO_L2_in_0_0,
        /* fifo */ fifo_A_A_IO_L2_in_0_1,
        /* fifo */ fifo_A_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_A_A_IO_L2_in_0_1,
        /* fifo */ fifo_A_A_IO_L2_in_0_2,
        /* fifo */ fifo_A_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_A_A_IO_L2_in_1_0,
        /* fifo */ fifo_A_A_IO_L2_in_1_1,
        /* fifo */ fifo_A_PE_2_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_A_A_IO_L2_in_1_1,
        /* fifo */ fifo_A_A_IO_L2_in_1_2,
        /* fifo */ fifo_A_PE_3_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L3_in(
        /* module id */ 0,
        /* array */ A,
        /* fifo */ fifo_A_A_L2_in_IO_0_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L3_in(
        /* module id */ 1,
        /* array */ A,
        /* fifo */ fifo_A_A_L2_in_IO_1_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_B_B_IO_L2_in_0_0,
        /* fifo */ fifo_B_B_IO_L2_in_0_1,
        /* fifo */ fifo_B_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_B_B_IO_L2_in_0_1,
        /* fifo */ fifo_B_B_IO_L2_in_0_2,
        /* fifo */ fifo_B_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_B_B_IO_L2_in_1_0,
        /* fifo */ fifo_B_B_IO_L2_in_1_1,
        /* fifo */ fifo_B_PE_0_2
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_B_B_IO_L2_in_1_1,
        /* fifo */ fifo_B_B_IO_L2_in_1_2,
        /* fifo */ fifo_B_PE_0_3
    );
    /* Module Call */

    /* Module Call */
    B_IO_L3_in(
        /* module id */ 0,
        /* array */ B,
        /* fifo */ fifo_B_B_L2_in_IO_0_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L3_in(
        /* module id */ 1,
        /* array */ B,
        /* fifo */ fifo_B_B_L2_in_IO_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_1,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_2,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* module id */ 2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_3,
        /* fifo */ fifo_C_drain_PE_2_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* module id */ 3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0_4,
        /* fifo */ fifo_C_drain_PE_3_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_2,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* module id */ 2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_3,
        /* fifo */ fifo_C_drain_PE_2_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* module id */ 3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1_4,
        /* fifo */ fifo_C_drain_PE_3_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_1,
        /* fifo */ fifo_C_drain_PE_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_2,
        /* fifo */ fifo_C_drain_PE_1_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* module id */ 2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_3,
        /* fifo */ fifo_C_drain_PE_2_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* module id */ 3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0_4,
        /* fifo */ fifo_C_drain_PE_3_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_1,
        /* fifo */ fifo_C_drain_PE_0_3
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_2,
        /* fifo */ fifo_C_drain_PE_1_3
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* module id */ 2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_2,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_3,
        /* fifo */ fifo_C_drain_PE_2_3
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* module id */ 3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_3,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1_4,
        /* fifo */ fifo_C_drain_PE_3_3
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_0_4
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0_2,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_1_4
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_0_4
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1_2,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_1_4
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L3_out(
        /* module id */ 0,
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L3_out(
        /* module id */ 1,
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_1_2
    );
    /* Module Call */

}
}
