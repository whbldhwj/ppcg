extern "C" {
void kernel0(A_t4 *A, B_t4 *B, C_t2 *C)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    /* FIFO Declaration */
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_0 depth=1
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_1 depth=1
    /* A_IO_L2_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_2 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_0;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_0 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_1;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_1 depth=1
    /* B_IO_L2_in fifo */ hls::stream<B_t4> fifo_B_B_IO_L2_in_2;
    #pragma HLS STREAM variable=fifo_B_B_IO_L2_in_2 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_0_0;
    #pragma HLS STREAM variable=fifo_A_PE_0_0 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_0_1;
    #pragma HLS STREAM variable=fifo_A_PE_0_1 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_0;
    #pragma HLS STREAM variable=fifo_A_PE_1_0 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_1;
    #pragma HLS STREAM variable=fifo_A_PE_1_1 depth=1
    /* PE fifo */ hls::stream<A_t2> fifo_A_PE_1_2;
    #pragma HLS STREAM variable=fifo_A_PE_1_2 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_0_0;
    #pragma HLS STREAM variable=fifo_B_PE_0_0 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_1_0;
    #pragma HLS STREAM variable=fifo_B_PE_1_0 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_0_1;
    #pragma HLS STREAM variable=fifo_B_PE_0_1 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_1_1;
    #pragma HLS STREAM variable=fifo_B_PE_1_1 depth=1
    /* PE fifo */ hls::stream<B_t2> fifo_B_PE_2_1;
    #pragma HLS STREAM variable=fifo_B_PE_2_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_0 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_0_1 depth=1
    /* PE fifo */ hls::stream<int> fifo_C_drain_PE_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_PE_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_0_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_0_2 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_0 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_1 depth=1
    /* C_drain_IO_L1_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L1_out_1_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L1_out_1_2 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_0;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_0 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_1;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_1 depth=1
    /* C_drain_IO_L2_out fifo */ hls::stream<C_t2> fifo_C_drain_C_drain_IO_L2_out_2;
    #pragma HLS STREAM variable=fifo_C_drain_C_drain_IO_L2_out_2 depth=1
    /* FIFO Declaration */

    /* Module Call */
    A_IO_L3_in(
        /* array */ A,
        /* fifo */ fifo_A_A_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 0,
        /* fifo */ fifo_A_A_IO_L2_in_0,
        /* fifo */ fifo_A_A_IO_L2_in_1,
        /* fifo */ fifo_A_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    A_IO_L2_in(
        /* module id */ 1,
        /* fifo */ fifo_A_A_IO_L2_in_1,
        /* fifo */ fifo_A_A_IO_L2_in_2,
        /* fifo */ fifo_A_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L3_in(
        /* array */ B,
        /* fifo */ fifo_B_B_L2_in_IO_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 0,
        /* fifo */ fifo_B_B_IO_L2_in_0,
        /* fifo */ fifo_B_B_IO_L2_in_1,
        /* fifo */ fifo_B_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    B_IO_L2_in(
        /* module id */ 1,
        /* fifo */ fifo_B_B_IO_L2_in_1,
        /* fifo */ fifo_B_B_IO_L2_in_2,
        /* fifo */ fifo_B_PE_0_1
    );
    /* Module Call */

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
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
        /* fifo */ fifo_C_drain_PE_0_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 0,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_0_2,
        /* fifo */ fifo_C_drain_PE_1_0
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
        /* fifo */ fifo_C_drain_PE_0_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L1_out(
        /* module id */ 1,
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L1_out_1_2,
        /* fifo */ fifo_C_drain_PE_1_1
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_0,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_0_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L2_out(
        /* module id */ 1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_1,
        /* fifo */ fifo_C_drain_C_drain_IO_L2_out_2,
        /* fifo */ fifo_C_drain_C_drain_L1_out_IO_1_2
    );
    /* Module Call */

    /* Module Call */
    C_drain_IO_L3_out(
        /* array */ C,
        /* fifo */ fifo_C_drain_C_drain_L2_out_IO_2
    );
    /* Module Call */

}
}
