`timescale 1 ns / 1 ps

module AESL_deadlock_detector (
    input reset,
    input clock);

    wire [2:0] proc_dep_vld_vec_0;
    reg [2:0] proc_dep_vld_vec_0_reg;
    wire [2:0] in_chan_dep_vld_vec_0;
    wire [65:0] in_chan_dep_data_vec_0;
    wire [2:0] token_in_vec_0;
    wire [2:0] out_chan_dep_vld_vec_0;
    wire [21:0] out_chan_dep_data_0;
    wire [2:0] token_out_vec_0;
    wire dl_detect_out_0;
    wire dep_chan_vld_1_0;
    wire [21:0] dep_chan_data_1_0;
    wire token_1_0;
    wire dep_chan_vld_4_0;
    wire [21:0] dep_chan_data_4_0;
    wire token_4_0;
    wire dep_chan_vld_21_0;
    wire [21:0] dep_chan_data_21_0;
    wire token_21_0;
    wire [2:0] proc_dep_vld_vec_1;
    reg [2:0] proc_dep_vld_vec_1_reg;
    wire [2:0] in_chan_dep_vld_vec_1;
    wire [65:0] in_chan_dep_data_vec_1;
    wire [2:0] token_in_vec_1;
    wire [2:0] out_chan_dep_vld_vec_1;
    wire [21:0] out_chan_dep_data_1;
    wire [2:0] token_out_vec_1;
    wire dl_detect_out_1;
    wire dep_chan_vld_0_1;
    wire [21:0] dep_chan_data_0_1;
    wire token_0_1;
    wire dep_chan_vld_2_1;
    wire [21:0] dep_chan_data_2_1;
    wire token_2_1;
    wire dep_chan_vld_4_1;
    wire [21:0] dep_chan_data_4_1;
    wire token_4_1;
    wire [2:0] proc_dep_vld_vec_2;
    reg [2:0] proc_dep_vld_vec_2_reg;
    wire [2:0] in_chan_dep_vld_vec_2;
    wire [65:0] in_chan_dep_data_vec_2;
    wire [2:0] token_in_vec_2;
    wire [2:0] out_chan_dep_vld_vec_2;
    wire [21:0] out_chan_dep_data_2;
    wire [2:0] token_out_vec_2;
    wire dl_detect_out_2;
    wire dep_chan_vld_1_2;
    wire [21:0] dep_chan_data_1_2;
    wire token_1_2;
    wire dep_chan_vld_3_2;
    wire [21:0] dep_chan_data_3_2;
    wire token_3_2;
    wire dep_chan_vld_7_2;
    wire [21:0] dep_chan_data_7_2;
    wire token_7_2;
    wire [1:0] proc_dep_vld_vec_3;
    reg [1:0] proc_dep_vld_vec_3_reg;
    wire [1:0] in_chan_dep_vld_vec_3;
    wire [43:0] in_chan_dep_data_vec_3;
    wire [1:0] token_in_vec_3;
    wire [1:0] out_chan_dep_vld_vec_3;
    wire [21:0] out_chan_dep_data_3;
    wire [1:0] token_out_vec_3;
    wire dl_detect_out_3;
    wire dep_chan_vld_2_3;
    wire [21:0] dep_chan_data_2_3;
    wire token_2_3;
    wire dep_chan_vld_10_3;
    wire [21:0] dep_chan_data_10_3;
    wire token_10_3;
    wire [2:0] proc_dep_vld_vec_4;
    reg [2:0] proc_dep_vld_vec_4_reg;
    wire [2:0] in_chan_dep_vld_vec_4;
    wire [65:0] in_chan_dep_data_vec_4;
    wire [2:0] token_in_vec_4;
    wire [2:0] out_chan_dep_vld_vec_4;
    wire [21:0] out_chan_dep_data_4;
    wire [2:0] token_out_vec_4;
    wire dl_detect_out_4;
    wire dep_chan_vld_0_4;
    wire [21:0] dep_chan_data_0_4;
    wire token_0_4;
    wire dep_chan_vld_1_4;
    wire [21:0] dep_chan_data_1_4;
    wire token_1_4;
    wire dep_chan_vld_5_4;
    wire [21:0] dep_chan_data_5_4;
    wire token_5_4;
    wire [2:0] proc_dep_vld_vec_5;
    reg [2:0] proc_dep_vld_vec_5_reg;
    wire [2:0] in_chan_dep_vld_vec_5;
    wire [65:0] in_chan_dep_data_vec_5;
    wire [2:0] token_in_vec_5;
    wire [2:0] out_chan_dep_vld_vec_5;
    wire [21:0] out_chan_dep_data_5;
    wire [2:0] token_out_vec_5;
    wire dl_detect_out_5;
    wire dep_chan_vld_4_5;
    wire [21:0] dep_chan_data_4_5;
    wire token_4_5;
    wire dep_chan_vld_6_5;
    wire [21:0] dep_chan_data_6_5;
    wire token_6_5;
    wire dep_chan_vld_7_5;
    wire [21:0] dep_chan_data_7_5;
    wire token_7_5;
    wire [1:0] proc_dep_vld_vec_6;
    reg [1:0] proc_dep_vld_vec_6_reg;
    wire [1:0] in_chan_dep_vld_vec_6;
    wire [43:0] in_chan_dep_data_vec_6;
    wire [1:0] token_in_vec_6;
    wire [1:0] out_chan_dep_vld_vec_6;
    wire [21:0] out_chan_dep_data_6;
    wire [1:0] token_out_vec_6;
    wire dl_detect_out_6;
    wire dep_chan_vld_5_6;
    wire [21:0] dep_chan_data_5_6;
    wire token_5_6;
    wire dep_chan_vld_8_6;
    wire [21:0] dep_chan_data_8_6;
    wire token_8_6;
    wire [4:0] proc_dep_vld_vec_7;
    reg [4:0] proc_dep_vld_vec_7_reg;
    wire [4:0] in_chan_dep_vld_vec_7;
    wire [109:0] in_chan_dep_data_vec_7;
    wire [4:0] token_in_vec_7;
    wire [4:0] out_chan_dep_vld_vec_7;
    wire [21:0] out_chan_dep_data_7;
    wire [4:0] token_out_vec_7;
    wire dl_detect_out_7;
    wire dep_chan_vld_2_7;
    wire [21:0] dep_chan_data_2_7;
    wire token_2_7;
    wire dep_chan_vld_5_7;
    wire [21:0] dep_chan_data_5_7;
    wire token_5_7;
    wire dep_chan_vld_8_7;
    wire [21:0] dep_chan_data_8_7;
    wire token_8_7;
    wire dep_chan_vld_10_7;
    wire [21:0] dep_chan_data_10_7;
    wire token_10_7;
    wire dep_chan_vld_15_7;
    wire [21:0] dep_chan_data_15_7;
    wire token_15_7;
    wire [4:0] proc_dep_vld_vec_8;
    reg [4:0] proc_dep_vld_vec_8_reg;
    wire [4:0] in_chan_dep_vld_vec_8;
    wire [109:0] in_chan_dep_data_vec_8;
    wire [4:0] token_in_vec_8;
    wire [4:0] out_chan_dep_vld_vec_8;
    wire [21:0] out_chan_dep_data_8;
    wire [4:0] token_out_vec_8;
    wire dl_detect_out_8;
    wire dep_chan_vld_6_8;
    wire [21:0] dep_chan_data_6_8;
    wire token_6_8;
    wire dep_chan_vld_7_8;
    wire [21:0] dep_chan_data_7_8;
    wire token_7_8;
    wire dep_chan_vld_9_8;
    wire [21:0] dep_chan_data_9_8;
    wire token_9_8;
    wire dep_chan_vld_12_8;
    wire [21:0] dep_chan_data_12_8;
    wire token_12_8;
    wire dep_chan_vld_17_8;
    wire [21:0] dep_chan_data_17_8;
    wire token_17_8;
    wire [0:0] proc_dep_vld_vec_9;
    reg [0:0] proc_dep_vld_vec_9_reg;
    wire [0:0] in_chan_dep_vld_vec_9;
    wire [21:0] in_chan_dep_data_vec_9;
    wire [0:0] token_in_vec_9;
    wire [0:0] out_chan_dep_vld_vec_9;
    wire [21:0] out_chan_dep_data_9;
    wire [0:0] token_out_vec_9;
    wire dl_detect_out_9;
    wire dep_chan_vld_8_9;
    wire [21:0] dep_chan_data_8_9;
    wire token_8_9;
    wire [4:0] proc_dep_vld_vec_10;
    reg [4:0] proc_dep_vld_vec_10_reg;
    wire [4:0] in_chan_dep_vld_vec_10;
    wire [109:0] in_chan_dep_data_vec_10;
    wire [4:0] token_in_vec_10;
    wire [4:0] out_chan_dep_vld_vec_10;
    wire [21:0] out_chan_dep_data_10;
    wire [4:0] token_out_vec_10;
    wire dl_detect_out_10;
    wire dep_chan_vld_3_10;
    wire [21:0] dep_chan_data_3_10;
    wire token_3_10;
    wire dep_chan_vld_7_10;
    wire [21:0] dep_chan_data_7_10;
    wire token_7_10;
    wire dep_chan_vld_11_10;
    wire [21:0] dep_chan_data_11_10;
    wire token_11_10;
    wire dep_chan_vld_12_10;
    wire [21:0] dep_chan_data_12_10;
    wire token_12_10;
    wire dep_chan_vld_16_10;
    wire [21:0] dep_chan_data_16_10;
    wire token_16_10;
    wire [0:0] proc_dep_vld_vec_11;
    reg [0:0] proc_dep_vld_vec_11_reg;
    wire [0:0] in_chan_dep_vld_vec_11;
    wire [21:0] in_chan_dep_data_vec_11;
    wire [0:0] token_in_vec_11;
    wire [0:0] out_chan_dep_vld_vec_11;
    wire [21:0] out_chan_dep_data_11;
    wire [0:0] token_out_vec_11;
    wire dl_detect_out_11;
    wire dep_chan_vld_10_11;
    wire [21:0] dep_chan_data_10_11;
    wire token_10_11;
    wire [4:0] proc_dep_vld_vec_12;
    reg [4:0] proc_dep_vld_vec_12_reg;
    wire [4:0] in_chan_dep_vld_vec_12;
    wire [109:0] in_chan_dep_data_vec_12;
    wire [4:0] token_in_vec_12;
    wire [4:0] out_chan_dep_vld_vec_12;
    wire [21:0] out_chan_dep_data_12;
    wire [4:0] token_out_vec_12;
    wire dl_detect_out_12;
    wire dep_chan_vld_8_12;
    wire [21:0] dep_chan_data_8_12;
    wire token_8_12;
    wire dep_chan_vld_10_12;
    wire [21:0] dep_chan_data_10_12;
    wire token_10_12;
    wire dep_chan_vld_13_12;
    wire [21:0] dep_chan_data_13_12;
    wire token_13_12;
    wire dep_chan_vld_14_12;
    wire [21:0] dep_chan_data_14_12;
    wire token_14_12;
    wire dep_chan_vld_18_12;
    wire [21:0] dep_chan_data_18_12;
    wire token_18_12;
    wire [0:0] proc_dep_vld_vec_13;
    reg [0:0] proc_dep_vld_vec_13_reg;
    wire [0:0] in_chan_dep_vld_vec_13;
    wire [21:0] in_chan_dep_data_vec_13;
    wire [0:0] token_in_vec_13;
    wire [0:0] out_chan_dep_vld_vec_13;
    wire [21:0] out_chan_dep_data_13;
    wire [0:0] token_out_vec_13;
    wire dl_detect_out_13;
    wire dep_chan_vld_12_13;
    wire [21:0] dep_chan_data_12_13;
    wire token_12_13;
    wire [0:0] proc_dep_vld_vec_14;
    reg [0:0] proc_dep_vld_vec_14_reg;
    wire [0:0] in_chan_dep_vld_vec_14;
    wire [21:0] in_chan_dep_data_vec_14;
    wire [0:0] token_in_vec_14;
    wire [0:0] out_chan_dep_vld_vec_14;
    wire [21:0] out_chan_dep_data_14;
    wire [0:0] token_out_vec_14;
    wire dl_detect_out_14;
    wire dep_chan_vld_12_14;
    wire [21:0] dep_chan_data_12_14;
    wire token_12_14;
    wire [1:0] proc_dep_vld_vec_15;
    reg [1:0] proc_dep_vld_vec_15_reg;
    wire [1:0] in_chan_dep_vld_vec_15;
    wire [43:0] in_chan_dep_data_vec_15;
    wire [1:0] token_in_vec_15;
    wire [1:0] out_chan_dep_vld_vec_15;
    wire [21:0] out_chan_dep_data_15;
    wire [1:0] token_out_vec_15;
    wire dl_detect_out_15;
    wire dep_chan_vld_7_15;
    wire [21:0] dep_chan_data_7_15;
    wire token_7_15;
    wire dep_chan_vld_16_15;
    wire [21:0] dep_chan_data_16_15;
    wire token_16_15;
    wire [2:0] proc_dep_vld_vec_16;
    reg [2:0] proc_dep_vld_vec_16_reg;
    wire [2:0] in_chan_dep_vld_vec_16;
    wire [65:0] in_chan_dep_data_vec_16;
    wire [2:0] token_in_vec_16;
    wire [2:0] out_chan_dep_vld_vec_16;
    wire [21:0] out_chan_dep_data_16;
    wire [2:0] token_out_vec_16;
    wire dl_detect_out_16;
    wire dep_chan_vld_10_16;
    wire [21:0] dep_chan_data_10_16;
    wire token_10_16;
    wire dep_chan_vld_15_16;
    wire [21:0] dep_chan_data_15_16;
    wire token_15_16;
    wire dep_chan_vld_19_16;
    wire [21:0] dep_chan_data_19_16;
    wire token_19_16;
    wire [1:0] proc_dep_vld_vec_17;
    reg [1:0] proc_dep_vld_vec_17_reg;
    wire [1:0] in_chan_dep_vld_vec_17;
    wire [43:0] in_chan_dep_data_vec_17;
    wire [1:0] token_in_vec_17;
    wire [1:0] out_chan_dep_vld_vec_17;
    wire [21:0] out_chan_dep_data_17;
    wire [1:0] token_out_vec_17;
    wire dl_detect_out_17;
    wire dep_chan_vld_8_17;
    wire [21:0] dep_chan_data_8_17;
    wire token_8_17;
    wire dep_chan_vld_18_17;
    wire [21:0] dep_chan_data_18_17;
    wire token_18_17;
    wire [2:0] proc_dep_vld_vec_18;
    reg [2:0] proc_dep_vld_vec_18_reg;
    wire [2:0] in_chan_dep_vld_vec_18;
    wire [65:0] in_chan_dep_data_vec_18;
    wire [2:0] token_in_vec_18;
    wire [2:0] out_chan_dep_vld_vec_18;
    wire [21:0] out_chan_dep_data_18;
    wire [2:0] token_out_vec_18;
    wire dl_detect_out_18;
    wire dep_chan_vld_12_18;
    wire [21:0] dep_chan_data_12_18;
    wire token_12_18;
    wire dep_chan_vld_17_18;
    wire [21:0] dep_chan_data_17_18;
    wire token_17_18;
    wire dep_chan_vld_20_18;
    wire [21:0] dep_chan_data_20_18;
    wire token_20_18;
    wire [1:0] proc_dep_vld_vec_19;
    reg [1:0] proc_dep_vld_vec_19_reg;
    wire [1:0] in_chan_dep_vld_vec_19;
    wire [43:0] in_chan_dep_data_vec_19;
    wire [1:0] token_in_vec_19;
    wire [1:0] out_chan_dep_vld_vec_19;
    wire [21:0] out_chan_dep_data_19;
    wire [1:0] token_out_vec_19;
    wire dl_detect_out_19;
    wire dep_chan_vld_16_19;
    wire [21:0] dep_chan_data_16_19;
    wire token_16_19;
    wire dep_chan_vld_20_19;
    wire [21:0] dep_chan_data_20_19;
    wire token_20_19;
    wire [2:0] proc_dep_vld_vec_20;
    reg [2:0] proc_dep_vld_vec_20_reg;
    wire [2:0] in_chan_dep_vld_vec_20;
    wire [65:0] in_chan_dep_data_vec_20;
    wire [2:0] token_in_vec_20;
    wire [2:0] out_chan_dep_vld_vec_20;
    wire [21:0] out_chan_dep_data_20;
    wire [2:0] token_out_vec_20;
    wire dl_detect_out_20;
    wire dep_chan_vld_18_20;
    wire [21:0] dep_chan_data_18_20;
    wire token_18_20;
    wire dep_chan_vld_19_20;
    wire [21:0] dep_chan_data_19_20;
    wire token_19_20;
    wire dep_chan_vld_21_20;
    wire [21:0] dep_chan_data_21_20;
    wire token_21_20;
    wire [1:0] proc_dep_vld_vec_21;
    reg [1:0] proc_dep_vld_vec_21_reg;
    wire [1:0] in_chan_dep_vld_vec_21;
    wire [43:0] in_chan_dep_data_vec_21;
    wire [1:0] token_in_vec_21;
    wire [1:0] out_chan_dep_vld_vec_21;
    wire [21:0] out_chan_dep_data_21;
    wire [1:0] token_out_vec_21;
    wire dl_detect_out_21;
    wire dep_chan_vld_0_21;
    wire [21:0] dep_chan_data_0_21;
    wire token_0_21;
    wire dep_chan_vld_20_21;
    wire [21:0] dep_chan_data_20_21;
    wire token_20_21;
    wire [21:0] dl_in_vec;
    wire dl_detect_out;
    wire [21:0] origin;
    wire token_clear;

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$kernel0_entry6_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$kernel0_entry6_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$kernel0_entry6_U0$ap_idle <= AESL_inst_kernel0.kernel0_entry6_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.kernel0_entry6_U0
    AESL_deadlock_detect_unit #(22, 0, 3, 3) AESL_deadlock_detect_unit_0 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_0),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_0),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_0),
        .token_in_vec(token_in_vec_0),
        .dl_detect_in(dl_detect_out),
        .origin(origin[0]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_0),
        .out_chan_dep_data(out_chan_dep_data_0),
        .token_out_vec(token_out_vec_0),
        .dl_detect_out(dl_in_vec[0]));

    assign proc_dep_vld_vec_0[0] = dl_detect_out ? proc_dep_vld_vec_0_reg[0] : (~AESL_inst_kernel0.kernel0_entry6_U0.A_V_out_blk_n | ((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0])));
    assign proc_dep_vld_vec_0[1] = dl_detect_out ? proc_dep_vld_vec_0_reg[1] : (~AESL_inst_kernel0.kernel0_entry6_U0.B_V_out_blk_n | ((AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0]) & AESL_inst_kernel0.kernel0_entry6_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0])));
    assign proc_dep_vld_vec_0[2] = dl_detect_out ? proc_dep_vld_vec_0_reg[2] : (~AESL_inst_kernel0.kernel0_entry6_U0.C_V_out_blk_n | (~AESL_inst_kernel0.start_for_C_drainrcU_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_0_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_0_reg <= proc_dep_vld_vec_0;
        end
    end
    assign in_chan_dep_vld_vec_0[0] = dep_chan_vld_1_0;
    assign in_chan_dep_data_vec_0[21 : 0] = dep_chan_data_1_0;
    assign token_in_vec_0[0] = token_1_0;
    assign in_chan_dep_vld_vec_0[1] = dep_chan_vld_4_0;
    assign in_chan_dep_data_vec_0[43 : 22] = dep_chan_data_4_0;
    assign token_in_vec_0[1] = token_4_0;
    assign in_chan_dep_vld_vec_0[2] = dep_chan_vld_21_0;
    assign in_chan_dep_data_vec_0[65 : 44] = dep_chan_data_21_0;
    assign token_in_vec_0[2] = token_21_0;
    assign dep_chan_vld_0_1 = out_chan_dep_vld_vec_0[0];
    assign dep_chan_data_0_1 = out_chan_dep_data_0;
    assign token_0_1 = token_out_vec_0[0];
    assign dep_chan_vld_0_4 = out_chan_dep_vld_vec_0[1];
    assign dep_chan_data_0_4 = out_chan_dep_data_0;
    assign token_0_4 = token_out_vec_0[1];
    assign dep_chan_vld_0_21 = out_chan_dep_vld_vec_0[2];
    assign dep_chan_data_0_21 = out_chan_dep_data_0;
    assign token_0_21 = token_out_vec_0[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$A_IO_L3_in_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$A_IO_L3_in_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$A_IO_L3_in_U0$ap_idle <= AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.A_IO_L3_in_U0
    AESL_deadlock_detect_unit #(22, 1, 3, 3) AESL_deadlock_detect_unit_1 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_1),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_1),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_1),
        .token_in_vec(token_in_vec_1),
        .dl_detect_in(dl_detect_out),
        .origin(origin[1]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_1),
        .out_chan_dep_data(out_chan_dep_data_1),
        .token_out_vec(token_out_vec_1),
        .dl_detect_out(dl_in_vec[1]));

    assign proc_dep_vld_vec_1[0] = dl_detect_out ? proc_dep_vld_vec_1_reg[0] : (~AESL_inst_kernel0.A_IO_L3_in_U0.A_V_offset_blk_n | ((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0])));
    assign proc_dep_vld_vec_1[1] = dl_detect_out ? proc_dep_vld_vec_1_reg[1] : (~AESL_inst_kernel0.A_IO_L3_in_U0.fifo_A_local_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_U0.ap_done));
    assign proc_dep_vld_vec_1[2] = dl_detect_out ? proc_dep_vld_vec_1_reg[2] : (((AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.A_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0])));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_1_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_1_reg <= proc_dep_vld_vec_1;
        end
    end
    assign in_chan_dep_vld_vec_1[0] = dep_chan_vld_0_1;
    assign in_chan_dep_data_vec_1[21 : 0] = dep_chan_data_0_1;
    assign token_in_vec_1[0] = token_0_1;
    assign in_chan_dep_vld_vec_1[1] = dep_chan_vld_2_1;
    assign in_chan_dep_data_vec_1[43 : 22] = dep_chan_data_2_1;
    assign token_in_vec_1[1] = token_2_1;
    assign in_chan_dep_vld_vec_1[2] = dep_chan_vld_4_1;
    assign in_chan_dep_data_vec_1[65 : 44] = dep_chan_data_4_1;
    assign token_in_vec_1[2] = token_4_1;
    assign dep_chan_vld_1_0 = out_chan_dep_vld_vec_1[0];
    assign dep_chan_data_1_0 = out_chan_dep_data_1;
    assign token_1_0 = token_out_vec_1[0];
    assign dep_chan_vld_1_2 = out_chan_dep_vld_vec_1[1];
    assign dep_chan_data_1_2 = out_chan_dep_data_1;
    assign token_1_2 = token_out_vec_1[1];
    assign dep_chan_vld_1_4 = out_chan_dep_vld_vec_1[2];
    assign dep_chan_data_1_4 = out_chan_dep_data_1;
    assign token_1_4 = token_out_vec_1[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$A_IO_L2_in_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$A_IO_L2_in_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$A_IO_L2_in_U0$ap_idle <= AESL_inst_kernel0.A_IO_L2_in_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.A_IO_L2_in_U0
    AESL_deadlock_detect_unit #(22, 2, 3, 3) AESL_deadlock_detect_unit_2 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_2),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_2),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_2),
        .token_in_vec(token_in_vec_2),
        .dl_detect_in(dl_detect_out),
        .origin(origin[2]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_2),
        .out_chan_dep_data(out_chan_dep_data_2),
        .token_out_vec(token_out_vec_2),
        .dl_detect_out(dl_in_vec[2]));

    assign proc_dep_vld_vec_2[0] = dl_detect_out ? proc_dep_vld_vec_2_reg[0] : (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_A_IO_L2sc4_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_U0.ap_ready | AESL_inst_kernel0$A_IO_L2_in_U0$ap_idle)));
    assign proc_dep_vld_vec_2[1] = dl_detect_out ? proc_dep_vld_vec_2_reg[1] : (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139.fifo_A_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_full_n & AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_done));
    assign proc_dep_vld_vec_2[2] = dl_detect_out ? proc_dep_vld_vec_2_reg[2] : (~AESL_inst_kernel0.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131.fifo_A_local_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE139_U0_U.if_full_n & AESL_inst_kernel0.PE139_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_2_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_2_reg <= proc_dep_vld_vec_2;
        end
    end
    assign in_chan_dep_vld_vec_2[0] = dep_chan_vld_1_2;
    assign in_chan_dep_data_vec_2[21 : 0] = dep_chan_data_1_2;
    assign token_in_vec_2[0] = token_1_2;
    assign in_chan_dep_vld_vec_2[1] = dep_chan_vld_3_2;
    assign in_chan_dep_data_vec_2[43 : 22] = dep_chan_data_3_2;
    assign token_in_vec_2[1] = token_3_2;
    assign in_chan_dep_vld_vec_2[2] = dep_chan_vld_7_2;
    assign in_chan_dep_data_vec_2[65 : 44] = dep_chan_data_7_2;
    assign token_in_vec_2[2] = token_7_2;
    assign dep_chan_vld_2_1 = out_chan_dep_vld_vec_2[0];
    assign dep_chan_data_2_1 = out_chan_dep_data_2;
    assign token_2_1 = token_out_vec_2[0];
    assign dep_chan_vld_2_3 = out_chan_dep_vld_vec_2[1];
    assign dep_chan_data_2_3 = out_chan_dep_data_2;
    assign token_2_3 = token_out_vec_2[1];
    assign dep_chan_vld_2_7 = out_chan_dep_vld_vec_2[2];
    assign dep_chan_data_2_7 = out_chan_dep_data_2;
    assign token_2_7 = token_out_vec_2[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$A_IO_L2_in_tail_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$A_IO_L2_in_tail_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$A_IO_L2_in_tail_U0$ap_idle <= AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.A_IO_L2_in_tail_U0
    AESL_deadlock_detect_unit #(22, 3, 2, 2) AESL_deadlock_detect_unit_3 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_3),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_3),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_3),
        .token_in_vec(token_in_vec_3),
        .dl_detect_in(dl_detect_out),
        .origin(origin[3]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_3),
        .out_chan_dep_data(out_chan_dep_data_3),
        .token_out_vec(token_out_vec_3),
        .dl_detect_out(dl_in_vec[3]));

    assign proc_dep_vld_vec_3[0] = dl_detect_out ? proc_dep_vld_vec_3_reg[0] : (~AESL_inst_kernel0.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_A_IO_L2tde_U.if_empty_n & (AESL_inst_kernel0.A_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0$A_IO_L2_in_tail_U0$ap_idle)));
    assign proc_dep_vld_vec_3[1] = dl_detect_out ? proc_dep_vld_vec_3_reg[1] : (~AESL_inst_kernel0.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117.fifo_A_local_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE142_U0_U.if_full_n & AESL_inst_kernel0.PE142_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_3_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_3_reg <= proc_dep_vld_vec_3;
        end
    end
    assign in_chan_dep_vld_vec_3[0] = dep_chan_vld_2_3;
    assign in_chan_dep_data_vec_3[21 : 0] = dep_chan_data_2_3;
    assign token_in_vec_3[0] = token_2_3;
    assign in_chan_dep_vld_vec_3[1] = dep_chan_vld_10_3;
    assign in_chan_dep_data_vec_3[43 : 22] = dep_chan_data_10_3;
    assign token_in_vec_3[1] = token_10_3;
    assign dep_chan_vld_3_2 = out_chan_dep_vld_vec_3[0];
    assign dep_chan_data_3_2 = out_chan_dep_data_3;
    assign token_3_2 = token_out_vec_3[0];
    assign dep_chan_vld_3_10 = out_chan_dep_vld_vec_3[1];
    assign dep_chan_data_3_10 = out_chan_dep_data_3;
    assign token_3_10 = token_out_vec_3[1];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$B_IO_L3_in_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$B_IO_L3_in_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$B_IO_L3_in_U0$ap_idle <= AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.B_IO_L3_in_U0
    AESL_deadlock_detect_unit #(22, 4, 3, 3) AESL_deadlock_detect_unit_4 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_4),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_4),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_4),
        .token_in_vec(token_in_vec_4),
        .dl_detect_in(dl_detect_out),
        .origin(origin[4]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_4),
        .out_chan_dep_data(out_chan_dep_data_4),
        .token_out_vec(token_out_vec_4),
        .dl_detect_out(dl_in_vec[4]));

    assign proc_dep_vld_vec_4[0] = dl_detect_out ? proc_dep_vld_vec_4_reg[0] : (~AESL_inst_kernel0.B_IO_L3_in_U0.B_V_offset_blk_n | ((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.kernel0_entry6_U0_ap_ready_count[0])));
    assign proc_dep_vld_vec_4[1] = dl_detect_out ? proc_dep_vld_vec_4_reg[1] : (~AESL_inst_kernel0.B_IO_L3_in_U0.fifo_B_local_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_U0.ap_done));
    assign proc_dep_vld_vec_4[2] = dl_detect_out ? proc_dep_vld_vec_4_reg[2] : (((AESL_inst_kernel0.B_IO_L3_in_U0_ap_ready_count[0]) & AESL_inst_kernel0.B_IO_L3_in_U0.ap_idle & ~(AESL_inst_kernel0.A_IO_L3_in_U0_ap_ready_count[0])));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_4_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_4_reg <= proc_dep_vld_vec_4;
        end
    end
    assign in_chan_dep_vld_vec_4[0] = dep_chan_vld_0_4;
    assign in_chan_dep_data_vec_4[21 : 0] = dep_chan_data_0_4;
    assign token_in_vec_4[0] = token_0_4;
    assign in_chan_dep_vld_vec_4[1] = dep_chan_vld_1_4;
    assign in_chan_dep_data_vec_4[43 : 22] = dep_chan_data_1_4;
    assign token_in_vec_4[1] = token_1_4;
    assign in_chan_dep_vld_vec_4[2] = dep_chan_vld_5_4;
    assign in_chan_dep_data_vec_4[65 : 44] = dep_chan_data_5_4;
    assign token_in_vec_4[2] = token_5_4;
    assign dep_chan_vld_4_0 = out_chan_dep_vld_vec_4[0];
    assign dep_chan_data_4_0 = out_chan_dep_data_4;
    assign token_4_0 = token_out_vec_4[0];
    assign dep_chan_vld_4_5 = out_chan_dep_vld_vec_4[1];
    assign dep_chan_data_4_5 = out_chan_dep_data_4;
    assign token_4_5 = token_out_vec_4[1];
    assign dep_chan_vld_4_1 = out_chan_dep_vld_vec_4[2];
    assign dep_chan_data_4_1 = out_chan_dep_data_4;
    assign token_4_1 = token_out_vec_4[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$B_IO_L2_in_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$B_IO_L2_in_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$B_IO_L2_in_U0$ap_idle <= AESL_inst_kernel0.B_IO_L2_in_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.B_IO_L2_in_U0
    AESL_deadlock_detect_unit #(22, 5, 3, 3) AESL_deadlock_detect_unit_5 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_5),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_5),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_5),
        .token_in_vec(token_in_vec_5),
        .dl_detect_in(dl_detect_out),
        .origin(origin[5]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_5),
        .out_chan_dep_data(out_chan_dep_data_5),
        .token_out_vec(token_out_vec_5),
        .dl_detect_out(dl_in_vec[5]));

    assign proc_dep_vld_vec_5[0] = dl_detect_out ? proc_dep_vld_vec_5_reg[0] : (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_B_IO_L2udo_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_U0.ap_ready | AESL_inst_kernel0$B_IO_L2_in_U0$ap_idle)));
    assign proc_dep_vld_vec_5[1] = dl_detect_out ? proc_dep_vld_vec_5_reg[1] : (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139.fifo_B_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_full_n & AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_done));
    assign proc_dep_vld_vec_5[2] = dl_detect_out ? proc_dep_vld_vec_5_reg[2] : (~AESL_inst_kernel0.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131.fifo_B_local_out_V_V_blk_n);
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_5_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_5_reg <= proc_dep_vld_vec_5;
        end
    end
    assign in_chan_dep_vld_vec_5[0] = dep_chan_vld_4_5;
    assign in_chan_dep_data_vec_5[21 : 0] = dep_chan_data_4_5;
    assign token_in_vec_5[0] = token_4_5;
    assign in_chan_dep_vld_vec_5[1] = dep_chan_vld_6_5;
    assign in_chan_dep_data_vec_5[43 : 22] = dep_chan_data_6_5;
    assign token_in_vec_5[1] = token_6_5;
    assign in_chan_dep_vld_vec_5[2] = dep_chan_vld_7_5;
    assign in_chan_dep_data_vec_5[65 : 44] = dep_chan_data_7_5;
    assign token_in_vec_5[2] = token_7_5;
    assign dep_chan_vld_5_4 = out_chan_dep_vld_vec_5[0];
    assign dep_chan_data_5_4 = out_chan_dep_data_5;
    assign token_5_4 = token_out_vec_5[0];
    assign dep_chan_vld_5_6 = out_chan_dep_vld_vec_5[1];
    assign dep_chan_data_5_6 = out_chan_dep_data_5;
    assign token_5_6 = token_out_vec_5[1];
    assign dep_chan_vld_5_7 = out_chan_dep_vld_vec_5[2];
    assign dep_chan_data_5_7 = out_chan_dep_data_5;
    assign token_5_7 = token_out_vec_5[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$B_IO_L2_in_tail_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$B_IO_L2_in_tail_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$B_IO_L2_in_tail_U0$ap_idle <= AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.B_IO_L2_in_tail_U0
    AESL_deadlock_detect_unit #(22, 6, 2, 2) AESL_deadlock_detect_unit_6 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_6),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_6),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_6),
        .token_in_vec(token_in_vec_6),
        .dl_detect_in(dl_detect_out),
        .origin(origin[6]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_6),
        .out_chan_dep_data(out_chan_dep_data_6),
        .token_out_vec(token_out_vec_6),
        .dl_detect_out(dl_in_vec[6]));

    assign proc_dep_vld_vec_6[0] = dl_detect_out ? proc_dep_vld_vec_6_reg[0] : (~AESL_inst_kernel0.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_B_IO_L2vdy_U.if_empty_n & (AESL_inst_kernel0.B_IO_L2_in_tail_U0.ap_ready | AESL_inst_kernel0$B_IO_L2_in_tail_U0$ap_idle)));
    assign proc_dep_vld_vec_6[1] = dl_detect_out ? proc_dep_vld_vec_6_reg[1] : (~AESL_inst_kernel0.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117.fifo_B_local_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE140_U0_U.if_full_n & AESL_inst_kernel0.PE140_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_6_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_6_reg <= proc_dep_vld_vec_6;
        end
    end
    assign in_chan_dep_vld_vec_6[0] = dep_chan_vld_5_6;
    assign in_chan_dep_data_vec_6[21 : 0] = dep_chan_data_5_6;
    assign token_in_vec_6[0] = token_5_6;
    assign in_chan_dep_vld_vec_6[1] = dep_chan_vld_8_6;
    assign in_chan_dep_data_vec_6[43 : 22] = dep_chan_data_8_6;
    assign token_in_vec_6[1] = token_8_6;
    assign dep_chan_vld_6_5 = out_chan_dep_vld_vec_6[0];
    assign dep_chan_data_6_5 = out_chan_dep_data_6;
    assign token_6_5 = token_out_vec_6[0];
    assign dep_chan_vld_6_8 = out_chan_dep_vld_vec_6[1];
    assign dep_chan_data_6_8 = out_chan_dep_data_6;
    assign token_6_8 = token_out_vec_6[1];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE139_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE139_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE139_U0$ap_idle <= AESL_inst_kernel0.PE139_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE139_U0
    AESL_deadlock_detect_unit #(22, 7, 5, 5) AESL_deadlock_detect_unit_7 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_7),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_7),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_7),
        .token_in_vec(token_in_vec_7),
        .dl_detect_in(dl_detect_out),
        .origin(origin[7]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_7),
        .out_chan_dep_data(out_chan_dep_data_7),
        .token_out_vec(token_out_vec_7),
        .dl_detect_out(dl_in_vec[7]));

    assign proc_dep_vld_vec_7[0] = dl_detect_out ? proc_dep_vld_vec_7_reg[0] : (~AESL_inst_kernel0.PE139_U0.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE139_U0_U.if_empty_n & (AESL_inst_kernel0.PE139_U0.ap_ready | AESL_inst_kernel0$PE139_U0$ap_idle)));
    assign proc_dep_vld_vec_7[1] = dl_detect_out ? proc_dep_vld_vec_7_reg[1] : (~AESL_inst_kernel0.PE139_U0.fifo_A_out_V_V_blk_n);
    assign proc_dep_vld_vec_7[2] = dl_detect_out ? proc_dep_vld_vec_7_reg[2] : (~AESL_inst_kernel0.PE139_U0.fifo_B_in_V_V_blk_n);
    assign proc_dep_vld_vec_7[3] = dl_detect_out ? proc_dep_vld_vec_7_reg[3] : (~AESL_inst_kernel0.PE139_U0.fifo_B_out_V_V_blk_n);
    assign proc_dep_vld_vec_7[4] = dl_detect_out ? proc_dep_vld_vec_7_reg[4] : (~AESL_inst_kernel0.PE139_U0.fifo_C_drain_out_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainwdI_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_7_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_7_reg <= proc_dep_vld_vec_7;
        end
    end
    assign in_chan_dep_vld_vec_7[0] = dep_chan_vld_2_7;
    assign in_chan_dep_data_vec_7[21 : 0] = dep_chan_data_2_7;
    assign token_in_vec_7[0] = token_2_7;
    assign in_chan_dep_vld_vec_7[1] = dep_chan_vld_5_7;
    assign in_chan_dep_data_vec_7[43 : 22] = dep_chan_data_5_7;
    assign token_in_vec_7[1] = token_5_7;
    assign in_chan_dep_vld_vec_7[2] = dep_chan_vld_8_7;
    assign in_chan_dep_data_vec_7[65 : 44] = dep_chan_data_8_7;
    assign token_in_vec_7[2] = token_8_7;
    assign in_chan_dep_vld_vec_7[3] = dep_chan_vld_10_7;
    assign in_chan_dep_data_vec_7[87 : 66] = dep_chan_data_10_7;
    assign token_in_vec_7[3] = token_10_7;
    assign in_chan_dep_vld_vec_7[4] = dep_chan_vld_15_7;
    assign in_chan_dep_data_vec_7[109 : 88] = dep_chan_data_15_7;
    assign token_in_vec_7[4] = token_15_7;
    assign dep_chan_vld_7_2 = out_chan_dep_vld_vec_7[0];
    assign dep_chan_data_7_2 = out_chan_dep_data_7;
    assign token_7_2 = token_out_vec_7[0];
    assign dep_chan_vld_7_8 = out_chan_dep_vld_vec_7[1];
    assign dep_chan_data_7_8 = out_chan_dep_data_7;
    assign token_7_8 = token_out_vec_7[1];
    assign dep_chan_vld_7_5 = out_chan_dep_vld_vec_7[2];
    assign dep_chan_data_7_5 = out_chan_dep_data_7;
    assign token_7_5 = token_out_vec_7[2];
    assign dep_chan_vld_7_10 = out_chan_dep_vld_vec_7[3];
    assign dep_chan_data_7_10 = out_chan_dep_data_7;
    assign token_7_10 = token_out_vec_7[3];
    assign dep_chan_vld_7_15 = out_chan_dep_vld_vec_7[4];
    assign dep_chan_data_7_15 = out_chan_dep_data_7;
    assign token_7_15 = token_out_vec_7[4];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE140_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE140_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE140_U0$ap_idle <= AESL_inst_kernel0.PE140_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE140_U0
    AESL_deadlock_detect_unit #(22, 8, 5, 5) AESL_deadlock_detect_unit_8 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_8),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_8),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_8),
        .token_in_vec(token_in_vec_8),
        .dl_detect_in(dl_detect_out),
        .origin(origin[8]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_8),
        .out_chan_dep_data(out_chan_dep_data_8),
        .token_out_vec(token_out_vec_8),
        .dl_detect_out(dl_in_vec[8]));

    assign proc_dep_vld_vec_8[0] = dl_detect_out ? proc_dep_vld_vec_8_reg[0] : (~AESL_inst_kernel0.PE140_U0.fifo_A_in_V_V_blk_n);
    assign proc_dep_vld_vec_8[1] = dl_detect_out ? proc_dep_vld_vec_8_reg[1] : (~AESL_inst_kernel0.PE140_U0.fifo_A_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_full_n & AESL_inst_kernel0.PE_A_dummy141_U0.ap_done));
    assign proc_dep_vld_vec_8[2] = dl_detect_out ? proc_dep_vld_vec_8_reg[2] : (~AESL_inst_kernel0.PE140_U0.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE140_U0_U.if_empty_n & (AESL_inst_kernel0.PE140_U0.ap_ready | AESL_inst_kernel0$PE140_U0$ap_idle)));
    assign proc_dep_vld_vec_8[3] = dl_detect_out ? proc_dep_vld_vec_8_reg[3] : (~AESL_inst_kernel0.PE140_U0.fifo_B_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_U0_U.if_full_n & AESL_inst_kernel0.PE_U0.ap_done));
    assign proc_dep_vld_vec_8[4] = dl_detect_out ? proc_dep_vld_vec_8_reg[4] : (~AESL_inst_kernel0.PE140_U0.fifo_C_drain_out_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainyd2_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_8_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_8_reg <= proc_dep_vld_vec_8;
        end
    end
    assign in_chan_dep_vld_vec_8[0] = dep_chan_vld_6_8;
    assign in_chan_dep_data_vec_8[21 : 0] = dep_chan_data_6_8;
    assign token_in_vec_8[0] = token_6_8;
    assign in_chan_dep_vld_vec_8[1] = dep_chan_vld_7_8;
    assign in_chan_dep_data_vec_8[43 : 22] = dep_chan_data_7_8;
    assign token_in_vec_8[1] = token_7_8;
    assign in_chan_dep_vld_vec_8[2] = dep_chan_vld_9_8;
    assign in_chan_dep_data_vec_8[65 : 44] = dep_chan_data_9_8;
    assign token_in_vec_8[2] = token_9_8;
    assign in_chan_dep_vld_vec_8[3] = dep_chan_vld_12_8;
    assign in_chan_dep_data_vec_8[87 : 66] = dep_chan_data_12_8;
    assign token_in_vec_8[3] = token_12_8;
    assign in_chan_dep_vld_vec_8[4] = dep_chan_vld_17_8;
    assign in_chan_dep_data_vec_8[109 : 88] = dep_chan_data_17_8;
    assign token_in_vec_8[4] = token_17_8;
    assign dep_chan_vld_8_7 = out_chan_dep_vld_vec_8[0];
    assign dep_chan_data_8_7 = out_chan_dep_data_8;
    assign token_8_7 = token_out_vec_8[0];
    assign dep_chan_vld_8_9 = out_chan_dep_vld_vec_8[1];
    assign dep_chan_data_8_9 = out_chan_dep_data_8;
    assign token_8_9 = token_out_vec_8[1];
    assign dep_chan_vld_8_6 = out_chan_dep_vld_vec_8[2];
    assign dep_chan_data_8_6 = out_chan_dep_data_8;
    assign token_8_6 = token_out_vec_8[2];
    assign dep_chan_vld_8_12 = out_chan_dep_vld_vec_8[3];
    assign dep_chan_data_8_12 = out_chan_dep_data_8;
    assign token_8_12 = token_out_vec_8[3];
    assign dep_chan_vld_8_17 = out_chan_dep_vld_vec_8[4];
    assign dep_chan_data_8_17 = out_chan_dep_data_8;
    assign token_8_17 = token_out_vec_8[4];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE_A_dummy141_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE_A_dummy141_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE_A_dummy141_U0$ap_idle <= AESL_inst_kernel0.PE_A_dummy141_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE_A_dummy141_U0
    AESL_deadlock_detect_unit #(22, 9, 1, 1) AESL_deadlock_detect_unit_9 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_9),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_9),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_9),
        .token_in_vec(token_in_vec_9),
        .dl_detect_in(dl_detect_out),
        .origin(origin[9]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_9),
        .out_chan_dep_data(out_chan_dep_data_9),
        .token_out_vec(token_out_vec_9),
        .dl_detect_out(dl_in_vec[9]));

    assign proc_dep_vld_vec_9[0] = dl_detect_out ? proc_dep_vld_vec_9_reg[0] : (~AESL_inst_kernel0.PE_A_dummy141_U0.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_A_duxdS_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy141_U0.ap_ready | AESL_inst_kernel0$PE_A_dummy141_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_9_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_9_reg <= proc_dep_vld_vec_9;
        end
    end
    assign in_chan_dep_vld_vec_9[0] = dep_chan_vld_8_9;
    assign in_chan_dep_data_vec_9[21 : 0] = dep_chan_data_8_9;
    assign token_in_vec_9[0] = token_8_9;
    assign dep_chan_vld_9_8 = out_chan_dep_vld_vec_9[0];
    assign dep_chan_data_9_8 = out_chan_dep_data_9;
    assign token_9_8 = token_out_vec_9[0];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE142_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE142_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE142_U0$ap_idle <= AESL_inst_kernel0.PE142_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE142_U0
    AESL_deadlock_detect_unit #(22, 10, 5, 5) AESL_deadlock_detect_unit_10 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_10),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_10),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_10),
        .token_in_vec(token_in_vec_10),
        .dl_detect_in(dl_detect_out),
        .origin(origin[10]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_10),
        .out_chan_dep_data(out_chan_dep_data_10),
        .token_out_vec(token_out_vec_10),
        .dl_detect_out(dl_in_vec[10]));

    assign proc_dep_vld_vec_10[0] = dl_detect_out ? proc_dep_vld_vec_10_reg[0] : (~AESL_inst_kernel0.PE142_U0.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE142_U0_U.if_empty_n & (AESL_inst_kernel0.PE142_U0.ap_ready | AESL_inst_kernel0$PE142_U0$ap_idle)));
    assign proc_dep_vld_vec_10[1] = dl_detect_out ? proc_dep_vld_vec_10_reg[1] : (~AESL_inst_kernel0.PE142_U0.fifo_A_out_V_V_blk_n);
    assign proc_dep_vld_vec_10[2] = dl_detect_out ? proc_dep_vld_vec_10_reg[2] : (~AESL_inst_kernel0.PE142_U0.fifo_B_in_V_V_blk_n);
    assign proc_dep_vld_vec_10[3] = dl_detect_out ? proc_dep_vld_vec_10_reg[3] : (~AESL_inst_kernel0.PE142_U0.fifo_B_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_full_n & AESL_inst_kernel0.PE_B_dummy143_U0.ap_done));
    assign proc_dep_vld_vec_10[4] = dl_detect_out ? proc_dep_vld_vec_10_reg[4] : (~AESL_inst_kernel0.PE142_U0.fifo_C_drain_out_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainAem_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_10_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_10_reg <= proc_dep_vld_vec_10;
        end
    end
    assign in_chan_dep_vld_vec_10[0] = dep_chan_vld_3_10;
    assign in_chan_dep_data_vec_10[21 : 0] = dep_chan_data_3_10;
    assign token_in_vec_10[0] = token_3_10;
    assign in_chan_dep_vld_vec_10[1] = dep_chan_vld_7_10;
    assign in_chan_dep_data_vec_10[43 : 22] = dep_chan_data_7_10;
    assign token_in_vec_10[1] = token_7_10;
    assign in_chan_dep_vld_vec_10[2] = dep_chan_vld_11_10;
    assign in_chan_dep_data_vec_10[65 : 44] = dep_chan_data_11_10;
    assign token_in_vec_10[2] = token_11_10;
    assign in_chan_dep_vld_vec_10[3] = dep_chan_vld_12_10;
    assign in_chan_dep_data_vec_10[87 : 66] = dep_chan_data_12_10;
    assign token_in_vec_10[3] = token_12_10;
    assign in_chan_dep_vld_vec_10[4] = dep_chan_vld_16_10;
    assign in_chan_dep_data_vec_10[109 : 88] = dep_chan_data_16_10;
    assign token_in_vec_10[4] = token_16_10;
    assign dep_chan_vld_10_3 = out_chan_dep_vld_vec_10[0];
    assign dep_chan_data_10_3 = out_chan_dep_data_10;
    assign token_10_3 = token_out_vec_10[0];
    assign dep_chan_vld_10_12 = out_chan_dep_vld_vec_10[1];
    assign dep_chan_data_10_12 = out_chan_dep_data_10;
    assign token_10_12 = token_out_vec_10[1];
    assign dep_chan_vld_10_7 = out_chan_dep_vld_vec_10[2];
    assign dep_chan_data_10_7 = out_chan_dep_data_10;
    assign token_10_7 = token_out_vec_10[2];
    assign dep_chan_vld_10_11 = out_chan_dep_vld_vec_10[3];
    assign dep_chan_data_10_11 = out_chan_dep_data_10;
    assign token_10_11 = token_out_vec_10[3];
    assign dep_chan_vld_10_16 = out_chan_dep_vld_vec_10[4];
    assign dep_chan_data_10_16 = out_chan_dep_data_10;
    assign token_10_16 = token_out_vec_10[4];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE_B_dummy143_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE_B_dummy143_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE_B_dummy143_U0$ap_idle <= AESL_inst_kernel0.PE_B_dummy143_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE_B_dummy143_U0
    AESL_deadlock_detect_unit #(22, 11, 1, 1) AESL_deadlock_detect_unit_11 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_11),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_11),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_11),
        .token_in_vec(token_in_vec_11),
        .dl_detect_in(dl_detect_out),
        .origin(origin[11]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_11),
        .out_chan_dep_data(out_chan_dep_data_11),
        .token_out_vec(token_out_vec_11),
        .dl_detect_out(dl_in_vec[11]));

    assign proc_dep_vld_vec_11[0] = dl_detect_out ? proc_dep_vld_vec_11_reg[0] : (~AESL_inst_kernel0.PE_B_dummy143_U0.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_B_duzec_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy143_U0.ap_ready | AESL_inst_kernel0$PE_B_dummy143_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_11_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_11_reg <= proc_dep_vld_vec_11;
        end
    end
    assign in_chan_dep_vld_vec_11[0] = dep_chan_vld_10_11;
    assign in_chan_dep_data_vec_11[21 : 0] = dep_chan_data_10_11;
    assign token_in_vec_11[0] = token_10_11;
    assign dep_chan_vld_11_10 = out_chan_dep_vld_vec_11[0];
    assign dep_chan_data_11_10 = out_chan_dep_data_11;
    assign token_11_10 = token_out_vec_11[0];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE_U0$ap_idle <= AESL_inst_kernel0.PE_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE_U0
    AESL_deadlock_detect_unit #(22, 12, 5, 5) AESL_deadlock_detect_unit_12 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_12),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_12),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_12),
        .token_in_vec(token_in_vec_12),
        .dl_detect_in(dl_detect_out),
        .origin(origin[12]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_12),
        .out_chan_dep_data(out_chan_dep_data_12),
        .token_out_vec(token_out_vec_12),
        .dl_detect_out(dl_in_vec[12]));

    assign proc_dep_vld_vec_12[0] = dl_detect_out ? proc_dep_vld_vec_12_reg[0] : (~AESL_inst_kernel0.PE_U0.fifo_A_in_V_V_blk_n);
    assign proc_dep_vld_vec_12[1] = dl_detect_out ? proc_dep_vld_vec_12_reg[1] : (~AESL_inst_kernel0.PE_U0.fifo_A_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_full_n & AESL_inst_kernel0.PE_A_dummy_U0.ap_done));
    assign proc_dep_vld_vec_12[2] = dl_detect_out ? proc_dep_vld_vec_12_reg[2] : (~AESL_inst_kernel0.PE_U0.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_U0_U.if_empty_n & (AESL_inst_kernel0.PE_U0.ap_ready | AESL_inst_kernel0$PE_U0$ap_idle)));
    assign proc_dep_vld_vec_12[3] = dl_detect_out ? proc_dep_vld_vec_12_reg[3] : (~AESL_inst_kernel0.PE_U0.fifo_B_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_full_n & AESL_inst_kernel0.PE_B_dummy_U0.ap_done));
    assign proc_dep_vld_vec_12[4] = dl_detect_out ? proc_dep_vld_vec_12_reg[4] : (~AESL_inst_kernel0.PE_U0.fifo_C_drain_out_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_done));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_12_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_12_reg <= proc_dep_vld_vec_12;
        end
    end
    assign in_chan_dep_vld_vec_12[0] = dep_chan_vld_8_12;
    assign in_chan_dep_data_vec_12[21 : 0] = dep_chan_data_8_12;
    assign token_in_vec_12[0] = token_8_12;
    assign in_chan_dep_vld_vec_12[1] = dep_chan_vld_10_12;
    assign in_chan_dep_data_vec_12[43 : 22] = dep_chan_data_10_12;
    assign token_in_vec_12[1] = token_10_12;
    assign in_chan_dep_vld_vec_12[2] = dep_chan_vld_13_12;
    assign in_chan_dep_data_vec_12[65 : 44] = dep_chan_data_13_12;
    assign token_in_vec_12[2] = token_13_12;
    assign in_chan_dep_vld_vec_12[3] = dep_chan_vld_14_12;
    assign in_chan_dep_data_vec_12[87 : 66] = dep_chan_data_14_12;
    assign token_in_vec_12[3] = token_14_12;
    assign in_chan_dep_vld_vec_12[4] = dep_chan_vld_18_12;
    assign in_chan_dep_data_vec_12[109 : 88] = dep_chan_data_18_12;
    assign token_in_vec_12[4] = token_18_12;
    assign dep_chan_vld_12_10 = out_chan_dep_vld_vec_12[0];
    assign dep_chan_data_12_10 = out_chan_dep_data_12;
    assign token_12_10 = token_out_vec_12[0];
    assign dep_chan_vld_12_13 = out_chan_dep_vld_vec_12[1];
    assign dep_chan_data_12_13 = out_chan_dep_data_12;
    assign token_12_13 = token_out_vec_12[1];
    assign dep_chan_vld_12_8 = out_chan_dep_vld_vec_12[2];
    assign dep_chan_data_12_8 = out_chan_dep_data_12;
    assign token_12_8 = token_out_vec_12[2];
    assign dep_chan_vld_12_14 = out_chan_dep_vld_vec_12[3];
    assign dep_chan_data_12_14 = out_chan_dep_data_12;
    assign token_12_14 = token_out_vec_12[3];
    assign dep_chan_vld_12_18 = out_chan_dep_vld_vec_12[4];
    assign dep_chan_data_12_18 = out_chan_dep_data_12;
    assign token_12_18 = token_out_vec_12[4];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE_A_dummy_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE_A_dummy_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE_A_dummy_U0$ap_idle <= AESL_inst_kernel0.PE_A_dummy_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE_A_dummy_U0
    AESL_deadlock_detect_unit #(22, 13, 1, 1) AESL_deadlock_detect_unit_13 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_13),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_13),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_13),
        .token_in_vec(token_in_vec_13),
        .dl_detect_in(dl_detect_out),
        .origin(origin[13]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_13),
        .out_chan_dep_data(out_chan_dep_data_13),
        .token_out_vec(token_out_vec_13),
        .dl_detect_out(dl_in_vec[13]));

    assign proc_dep_vld_vec_13[0] = dl_detect_out ? proc_dep_vld_vec_13_reg[0] : (~AESL_inst_kernel0.PE_A_dummy_U0.fifo_A_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_A_duBew_U.if_empty_n & (AESL_inst_kernel0.PE_A_dummy_U0.ap_ready | AESL_inst_kernel0$PE_A_dummy_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_13_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_13_reg <= proc_dep_vld_vec_13;
        end
    end
    assign in_chan_dep_vld_vec_13[0] = dep_chan_vld_12_13;
    assign in_chan_dep_data_vec_13[21 : 0] = dep_chan_data_12_13;
    assign token_in_vec_13[0] = token_12_13;
    assign dep_chan_vld_13_12 = out_chan_dep_vld_vec_13[0];
    assign dep_chan_data_13_12 = out_chan_dep_data_13;
    assign token_13_12 = token_out_vec_13[0];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$PE_B_dummy_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$PE_B_dummy_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$PE_B_dummy_U0$ap_idle <= AESL_inst_kernel0.PE_B_dummy_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.PE_B_dummy_U0
    AESL_deadlock_detect_unit #(22, 14, 1, 1) AESL_deadlock_detect_unit_14 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_14),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_14),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_14),
        .token_in_vec(token_in_vec_14),
        .dl_detect_in(dl_detect_out),
        .origin(origin[14]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_14),
        .out_chan_dep_data(out_chan_dep_data_14),
        .token_out_vec(token_out_vec_14),
        .dl_detect_out(dl_in_vec[14]));

    assign proc_dep_vld_vec_14[0] = dl_detect_out ? proc_dep_vld_vec_14_reg[0] : (~AESL_inst_kernel0.PE_B_dummy_U0.fifo_B_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_PE_B_duCeG_U.if_empty_n & (AESL_inst_kernel0.PE_B_dummy_U0.ap_ready | AESL_inst_kernel0$PE_B_dummy_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_14_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_14_reg <= proc_dep_vld_vec_14;
        end
    end
    assign in_chan_dep_vld_vec_14[0] = dep_chan_vld_12_14;
    assign in_chan_dep_data_vec_14[21 : 0] = dep_chan_data_12_14;
    assign token_in_vec_14[0] = token_12_14;
    assign dep_chan_vld_14_12 = out_chan_dep_vld_vec_14[0];
    assign dep_chan_data_14_12 = out_chan_dep_data_14;
    assign token_14_12 = token_out_vec_14[0];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L1_out_he_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L1_out_he_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L1_out_he_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L1_out_he_U0
    AESL_deadlock_detect_unit #(22, 15, 2, 2) AESL_deadlock_detect_unit_15 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_15),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_15),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_15),
        .token_in_vec(token_in_vec_15),
        .dl_detect_in(dl_detect_out),
        .origin(origin[15]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_15),
        .out_chan_dep_data(out_chan_dep_data_15),
        .token_out_vec(token_out_vec_15),
        .dl_detect_out(dl_in_vec[15]));

    assign proc_dep_vld_vec_15[0] = dl_detect_out ? proc_dep_vld_vec_15_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113.fifo_C_drain_out_V_V_blk_n);
    assign proc_dep_vld_vec_15[1] = dl_detect_out ? proc_dep_vld_vec_15_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106.fifo_C_drain_local_in_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainwdI_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L1_out_he_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_15_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_15_reg <= proc_dep_vld_vec_15;
        end
    end
    assign in_chan_dep_vld_vec_15[0] = dep_chan_vld_7_15;
    assign in_chan_dep_data_vec_15[21 : 0] = dep_chan_data_7_15;
    assign token_in_vec_15[0] = token_7_15;
    assign in_chan_dep_vld_vec_15[1] = dep_chan_vld_16_15;
    assign in_chan_dep_data_vec_15[43 : 22] = dep_chan_data_16_15;
    assign token_in_vec_15[1] = token_16_15;
    assign dep_chan_vld_15_16 = out_chan_dep_vld_vec_15[0];
    assign dep_chan_data_15_16 = out_chan_dep_data_15;
    assign token_15_16 = token_out_vec_15[0];
    assign dep_chan_vld_15_7 = out_chan_dep_vld_vec_15[1];
    assign dep_chan_data_15_7 = out_chan_dep_data_15;
    assign token_15_7 = token_out_vec_15[1];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L1_out145_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L1_out145_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L1_out145_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L1_out145_U0
    AESL_deadlock_detect_unit #(22, 16, 3, 3) AESL_deadlock_detect_unit_16 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_16),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_16),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_16),
        .token_in_vec(token_in_vec_16),
        .dl_detect_in(dl_detect_out),
        .origin(origin[16]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_16),
        .out_chan_dep_data(out_chan_dep_data_16),
        .token_out_vec(token_out_vec_16),
        .dl_detect_out(dl_in_vec[16]));

    assign proc_dep_vld_vec_16[0] = dl_detect_out ? proc_dep_vld_vec_16_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_in_V_V_blk_n);
    assign proc_dep_vld_vec_16[1] = dl_detect_out ? proc_dep_vld_vec_16_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainEe0_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_done));
    assign proc_dep_vld_vec_16[2] = dl_detect_out ? proc_dep_vld_vec_16_reg[2] : (~AESL_inst_kernel0.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130.fifo_C_drain_local_in_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainAem_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out145_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L1_out145_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_16_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_16_reg <= proc_dep_vld_vec_16;
        end
    end
    assign in_chan_dep_vld_vec_16[0] = dep_chan_vld_10_16;
    assign in_chan_dep_data_vec_16[21 : 0] = dep_chan_data_10_16;
    assign token_in_vec_16[0] = token_10_16;
    assign in_chan_dep_vld_vec_16[1] = dep_chan_vld_15_16;
    assign in_chan_dep_data_vec_16[43 : 22] = dep_chan_data_15_16;
    assign token_in_vec_16[1] = token_15_16;
    assign in_chan_dep_vld_vec_16[2] = dep_chan_vld_19_16;
    assign in_chan_dep_data_vec_16[65 : 44] = dep_chan_data_19_16;
    assign token_in_vec_16[2] = token_19_16;
    assign dep_chan_vld_16_15 = out_chan_dep_vld_vec_16[0];
    assign dep_chan_data_16_15 = out_chan_dep_data_16;
    assign token_16_15 = token_out_vec_16[0];
    assign dep_chan_vld_16_19 = out_chan_dep_vld_vec_16[1];
    assign dep_chan_data_16_19 = out_chan_dep_data_16;
    assign token_16_19 = token_out_vec_16[1];
    assign dep_chan_vld_16_10 = out_chan_dep_vld_vec_16[2];
    assign dep_chan_data_16_10 = out_chan_dep_data_16;
    assign token_16_10 = token_out_vec_16[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L1_out_he_1_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L1_out_he_1_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L1_out_he_1_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0
    AESL_deadlock_detect_unit #(22, 17, 2, 2) AESL_deadlock_detect_unit_17 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_17),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_17),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_17),
        .token_in_vec(token_in_vec_17),
        .dl_detect_in(dl_detect_out),
        .origin(origin[17]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_17),
        .out_chan_dep_data(out_chan_dep_data_17),
        .token_out_vec(token_out_vec_17),
        .dl_detect_out(dl_in_vec[17]));

    assign proc_dep_vld_vec_17[0] = dl_detect_out ? proc_dep_vld_vec_17_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113.fifo_C_drain_out_V_V_blk_n);
    assign proc_dep_vld_vec_17[1] = dl_detect_out ? proc_dep_vld_vec_17_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106.fifo_C_drain_local_in_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainyd2_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_he_1_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L1_out_he_1_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_17_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_17_reg <= proc_dep_vld_vec_17;
        end
    end
    assign in_chan_dep_vld_vec_17[0] = dep_chan_vld_8_17;
    assign in_chan_dep_data_vec_17[21 : 0] = dep_chan_data_8_17;
    assign token_in_vec_17[0] = token_8_17;
    assign in_chan_dep_vld_vec_17[1] = dep_chan_vld_18_17;
    assign in_chan_dep_data_vec_17[43 : 22] = dep_chan_data_18_17;
    assign token_in_vec_17[1] = token_18_17;
    assign dep_chan_vld_17_18 = out_chan_dep_vld_vec_17[0];
    assign dep_chan_data_17_18 = out_chan_dep_data_17;
    assign token_17_18 = token_out_vec_17[0];
    assign dep_chan_vld_17_8 = out_chan_dep_vld_vec_17[1];
    assign dep_chan_data_17_8 = out_chan_dep_data_17;
    assign token_17_8 = token_out_vec_17[1];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L1_out_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L1_out_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L1_out_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L1_out_U0
    AESL_deadlock_detect_unit #(22, 18, 3, 3) AESL_deadlock_detect_unit_18 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_18),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_18),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_18),
        .token_in_vec(token_in_vec_18),
        .dl_detect_in(dl_detect_out),
        .origin(origin[18]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_18),
        .out_chan_dep_data(out_chan_dep_data_18),
        .token_out_vec(token_out_vec_18),
        .dl_detect_out(dl_in_vec[18]));

    assign proc_dep_vld_vec_18[0] = dl_detect_out ? proc_dep_vld_vec_18_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_in_V_V_blk_n);
    assign proc_dep_vld_vec_18[1] = dl_detect_out ? proc_dep_vld_vec_18_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120.fifo_C_drain_out_V_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainFfa_U.if_full_n & AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_done));
    assign proc_dep_vld_vec_18[2] = dl_detect_out ? proc_dep_vld_vec_18_reg[2] : (~AESL_inst_kernel0.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130.fifo_C_drain_local_in_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainDeQ_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L1_out_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L1_out_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_18_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_18_reg <= proc_dep_vld_vec_18;
        end
    end
    assign in_chan_dep_vld_vec_18[0] = dep_chan_vld_12_18;
    assign in_chan_dep_data_vec_18[21 : 0] = dep_chan_data_12_18;
    assign token_in_vec_18[0] = token_12_18;
    assign in_chan_dep_vld_vec_18[1] = dep_chan_vld_17_18;
    assign in_chan_dep_data_vec_18[43 : 22] = dep_chan_data_17_18;
    assign token_in_vec_18[1] = token_17_18;
    assign in_chan_dep_vld_vec_18[2] = dep_chan_vld_20_18;
    assign in_chan_dep_data_vec_18[65 : 44] = dep_chan_data_20_18;
    assign token_in_vec_18[2] = token_20_18;
    assign dep_chan_vld_18_17 = out_chan_dep_vld_vec_18[0];
    assign dep_chan_data_18_17 = out_chan_dep_data_18;
    assign token_18_17 = token_out_vec_18[0];
    assign dep_chan_vld_18_20 = out_chan_dep_vld_vec_18[1];
    assign dep_chan_data_18_20 = out_chan_dep_data_18;
    assign token_18_20 = token_out_vec_18[1];
    assign dep_chan_vld_18_12 = out_chan_dep_vld_vec_18[2];
    assign dep_chan_data_18_12 = out_chan_dep_data_18;
    assign token_18_12 = token_out_vec_18[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L2_out_he_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L2_out_he_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L2_out_he_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L2_out_he_U0
    AESL_deadlock_detect_unit #(22, 19, 2, 2) AESL_deadlock_detect_unit_19 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_19),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_19),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_19),
        .token_in_vec(token_in_vec_19),
        .dl_detect_in(dl_detect_out),
        .origin(origin[19]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_19),
        .out_chan_dep_data(out_chan_dep_data_19),
        .token_out_vec(token_out_vec_19),
        .dl_detect_out(dl_in_vec[19]));

    assign proc_dep_vld_vec_19[0] = dl_detect_out ? proc_dep_vld_vec_19_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.fifo_C_drain_out_V_V_blk_n);
    assign proc_dep_vld_vec_19[1] = dl_detect_out ? proc_dep_vld_vec_19_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.fifo_C_drain_local_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainEe0_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_he_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L2_out_he_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_19_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_19_reg <= proc_dep_vld_vec_19;
        end
    end
    assign in_chan_dep_vld_vec_19[0] = dep_chan_vld_16_19;
    assign in_chan_dep_data_vec_19[21 : 0] = dep_chan_data_16_19;
    assign token_in_vec_19[0] = token_16_19;
    assign in_chan_dep_vld_vec_19[1] = dep_chan_vld_20_19;
    assign in_chan_dep_data_vec_19[43 : 22] = dep_chan_data_20_19;
    assign token_in_vec_19[1] = token_20_19;
    assign dep_chan_vld_19_20 = out_chan_dep_vld_vec_19[0];
    assign dep_chan_data_19_20 = out_chan_dep_data_19;
    assign token_19_20 = token_out_vec_19[0];
    assign dep_chan_vld_19_16 = out_chan_dep_vld_vec_19[1];
    assign dep_chan_data_19_16 = out_chan_dep_data_19;
    assign token_19_16 = token_out_vec_19[1];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L2_out_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L2_out_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L2_out_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L2_out_U0
    AESL_deadlock_detect_unit #(22, 20, 3, 3) AESL_deadlock_detect_unit_20 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_20),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_20),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_20),
        .token_in_vec(token_in_vec_20),
        .dl_detect_in(dl_detect_out),
        .origin(origin[20]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_20),
        .out_chan_dep_data(out_chan_dep_data_20),
        .token_out_vec(token_out_vec_20),
        .dl_detect_out(dl_in_vec[20]));

    assign proc_dep_vld_vec_20[0] = dl_detect_out ? proc_dep_vld_vec_20_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_in_V_V_blk_n);
    assign proc_dep_vld_vec_20[1] = dl_detect_out ? proc_dep_vld_vec_20_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_out_V_V_blk_n);
    assign proc_dep_vld_vec_20[2] = dl_detect_out ? proc_dep_vld_vec_20_reg[2] : (~AESL_inst_kernel0.C_drain_IO_L2_out_U0.fifo_C_drain_local_in_V_V_blk_n | (~AESL_inst_kernel0.start_for_C_drainFfa_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L2_out_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L2_out_U0$ap_idle)));
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_20_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_20_reg <= proc_dep_vld_vec_20;
        end
    end
    assign in_chan_dep_vld_vec_20[0] = dep_chan_vld_18_20;
    assign in_chan_dep_data_vec_20[21 : 0] = dep_chan_data_18_20;
    assign token_in_vec_20[0] = token_18_20;
    assign in_chan_dep_vld_vec_20[1] = dep_chan_vld_19_20;
    assign in_chan_dep_data_vec_20[43 : 22] = dep_chan_data_19_20;
    assign token_in_vec_20[1] = token_19_20;
    assign in_chan_dep_vld_vec_20[2] = dep_chan_vld_21_20;
    assign in_chan_dep_data_vec_20[65 : 44] = dep_chan_data_21_20;
    assign token_in_vec_20[2] = token_21_20;
    assign dep_chan_vld_20_19 = out_chan_dep_vld_vec_20[0];
    assign dep_chan_data_20_19 = out_chan_dep_data_20;
    assign token_20_19 = token_out_vec_20[0];
    assign dep_chan_vld_20_21 = out_chan_dep_vld_vec_20[1];
    assign dep_chan_data_20_21 = out_chan_dep_data_20;
    assign token_20_21 = token_out_vec_20[1];
    assign dep_chan_vld_20_18 = out_chan_dep_vld_vec_20[2];
    assign dep_chan_data_20_18 = out_chan_dep_data_20;
    assign token_20_18 = token_out_vec_20[2];

    // delay ap_idle for one cycle
    reg [0:0] AESL_inst_kernel0$C_drain_IO_L3_out_U0$ap_idle;
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            AESL_inst_kernel0$C_drain_IO_L3_out_U0$ap_idle <= 'b0;
        end
        else begin
            AESL_inst_kernel0$C_drain_IO_L3_out_U0$ap_idle <= AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_idle;
        end
    end
    // Process: AESL_inst_kernel0.C_drain_IO_L3_out_U0
    AESL_deadlock_detect_unit #(22, 21, 2, 2) AESL_deadlock_detect_unit_21 (
        .reset(reset),
        .clock(clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_21),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_21),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_21),
        .token_in_vec(token_in_vec_21),
        .dl_detect_in(dl_detect_out),
        .origin(origin[21]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_21),
        .out_chan_dep_data(out_chan_dep_data_21),
        .token_out_vec(token_out_vec_21),
        .dl_detect_out(dl_in_vec[21]));

    assign proc_dep_vld_vec_21[0] = dl_detect_out ? proc_dep_vld_vec_21_reg[0] : (~AESL_inst_kernel0.C_drain_IO_L3_out_U0.C_V_offset_blk_n | (~AESL_inst_kernel0.start_for_C_drainrcU_U.if_empty_n & (AESL_inst_kernel0.C_drain_IO_L3_out_U0.ap_ready | AESL_inst_kernel0$C_drain_IO_L3_out_U0$ap_idle)));
    assign proc_dep_vld_vec_21[1] = dl_detect_out ? proc_dep_vld_vec_21_reg[1] : (~AESL_inst_kernel0.C_drain_IO_L3_out_U0.fifo_C_drain_local_in_V_V_blk_n);
    always @ (negedge reset or posedge clock) begin
        if (~reset) begin
            proc_dep_vld_vec_21_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_21_reg <= proc_dep_vld_vec_21;
        end
    end
    assign in_chan_dep_vld_vec_21[0] = dep_chan_vld_0_21;
    assign in_chan_dep_data_vec_21[21 : 0] = dep_chan_data_0_21;
    assign token_in_vec_21[0] = token_0_21;
    assign in_chan_dep_vld_vec_21[1] = dep_chan_vld_20_21;
    assign in_chan_dep_data_vec_21[43 : 22] = dep_chan_data_20_21;
    assign token_in_vec_21[1] = token_20_21;
    assign dep_chan_vld_21_0 = out_chan_dep_vld_vec_21[0];
    assign dep_chan_data_21_0 = out_chan_dep_data_21;
    assign token_21_0 = token_out_vec_21[0];
    assign dep_chan_vld_21_20 = out_chan_dep_vld_vec_21[1];
    assign dep_chan_data_21_20 = out_chan_dep_data_21;
    assign token_21_20 = token_out_vec_21[1];


    AESL_deadlock_report_unit #(22) AESL_deadlock_report_unit_inst (
        .reset(reset),
        .clock(clock),
        .dl_in_vec(dl_in_vec),
        .dl_detect_out(dl_detect_out),
        .origin(origin),
        .token_clear(token_clear));

endmodule
