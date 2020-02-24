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
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_1;
    /* PE fifo */ hls::stream<int> fifo_A_PE_1_2;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_0;
    /* PE fifo */ hls::stream<int> fifo_B_PE_0_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_1_1;
    /* PE fifo */ hls::stream<int> fifo_B_PE_2_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
    /* A_L2_in_IO fifo */ hls::stream<int> fifo_A_A_L2_in_IO_0;
    /* A_L2_in_IO fifo */ hls::stream<int> fifo_A_A_L2_in_IO_1;
    /* A_L2_in_IO fifo */ hls::stream<int> fifo_A_A_L2_in_IO_2;
    /* B_L2_in_IO fifo */ hls::stream<int> fifo_B_B_L2_in_IO_0;
    /* B_L2_in_IO fifo */ hls::stream<int> fifo_B_B_L2_in_IO_1;
    /* B_L2_in_IO fifo */ hls::stream<int> fifo_B_B_L2_in_IO_2;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_0;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_1;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_2;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_0;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_1;
    /* C_drain_L1_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_2;
    /* C_drain_L2_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_0;
    /* C_drain_L2_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_1;
    /* C_drain_L2_out_IO fifo */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_2;
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
    A_L2_in_IO(
        /* module id */ 0,
        /* fifo */ fifo_A_A_L2_in_IO_0,
        /* fifo */ fifo_A_A_L2_in_IO_1,
        /* fifo */ fifo_A_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    A_L2_in_IO(
        /* module id */ 1,
        /* fifo */ fifo_A_A_L2_in_IO_1,
        /* fifo */ fifo_A_A_L2_in_IO_2,
        /* fifo */ fifo_A_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    A_L3_in_IO(
        /* array */ A,
        /* fifo */ fifo_A_A_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    B_L2_in_IO(
        /* module id */ 0,
        /* fifo */ fifo_B_B_L2_in_IO_0,
        /* fifo */ fifo_B_B_L2_in_IO_1,
        /* fifo */ fifo_B_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    B_L2_in_IO(
        /* module id */ 1,
        /* fifo */ fifo_B_B_L2_in_IO_1,
        /* fifo */ fifo_B_B_L2_in_IO_2,
        /* fifo */ fifo_B_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    B_L3_in_IO(
        /* array */ B,
        /* fifo */ fifo_B_B_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_1,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_2,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_0,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_L1_out_IO(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_2,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_L2_out_IO(
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_0,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_L2_out_IO(
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_1,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_2,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_L3_out_IO(
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_2
    );
    /* Module Call */

}
}
