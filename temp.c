void kernel0(int *A, int *B, int *C)
{
    /* PE FIFO */ hls::stream<int> fifo_A_PE_0_0;
    /* PE FIFO */ hls::stream<int> fifo_A_PE_0_1;
    /* PE FIFO */ hls::stream<int> fifo_A_PE_0_2;
    /* PE FIFO */ hls::stream<int> fifo_A_PE_1_0;
    /* PE FIFO */ hls::stream<int> fifo_A_PE_1_1;
    /* PE FIFO */ hls::stream<int> fifo_A_PE_1_2;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_0_0;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_1_0;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_2_0;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_0_1;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_1_1;
    /* PE FIFO */ hls::stream<int> fifo_B_PE_2_1;
    /* PE FIFO */ hls::stream<int> fifo_C_drain_PE_0_0;
    /* PE FIFO */ hls::stream<int> fifo_C_drain_PE_1_0;
    /* PE FIFO */ hls::stream<int> fifo_C_drain_PE_0_1;
    /* PE FIFO */ hls::stream<int> fifo_C_drain_PE_1_1;
    /* A_L2_in_IO FIFO */ hls::stream<int> fifo_A_A_L2_in_IO_0;
    /* A_L2_in_IO FIFO */ hls::stream<int> fifo_A_A_L2_in_IO_1;
    /* A_L2_in_IO FIFO */ hls::stream<int> fifo_A_A_L2_in_IO_2;
    /* B_L2_in_IO FIFO */ hls::stream<int> fifo_B_B_L2_in_IO_0;
    /* B_L2_in_IO FIFO */ hls::stream<int> fifo_B_B_L2_in_IO_1;
    /* B_L2_in_IO FIFO */ hls::stream<int> fifo_B_B_L2_in_IO_2;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_0;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_1;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_0_2;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_0;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_1;
    /* C_drain_L1_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L1_out_IO_1_2;
    /* C_drain_L2_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_0;
    /* C_drain_L2_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_1;
    /* C_drain_L2_out_IO FIFO */ hls::stream<int> fifo_C_drain_C_drain_L2_out_IO_2;
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
