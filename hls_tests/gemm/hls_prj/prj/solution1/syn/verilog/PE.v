// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module PE (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        start_out,
        start_write,
        fifo_A_in_V_V_dout,
        fifo_A_in_V_V_empty_n,
        fifo_A_in_V_V_read,
        fifo_A_out_V_V_din,
        fifo_A_out_V_V_full_n,
        fifo_A_out_V_V_write,
        fifo_B_in_V_V_dout,
        fifo_B_in_V_V_empty_n,
        fifo_B_in_V_V_read,
        fifo_B_out_V_V_din,
        fifo_B_out_V_V_full_n,
        fifo_B_out_V_V_write,
        fifo_C_drain_out_V_din,
        fifo_C_drain_out_V_full_n,
        fifo_C_drain_out_V_write
);

parameter    ap_ST_fsm_state1 = 6'd1;
parameter    ap_ST_fsm_state2 = 6'd2;
parameter    ap_ST_fsm_state3 = 6'd4;
parameter    ap_ST_fsm_state4 = 6'd8;
parameter    ap_ST_fsm_pp1_stage0 = 6'd16;
parameter    ap_ST_fsm_state9 = 6'd32;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   start_out;
output   start_write;
input  [63:0] fifo_A_in_V_V_dout;
input   fifo_A_in_V_V_empty_n;
output   fifo_A_in_V_V_read;
output  [63:0] fifo_A_out_V_V_din;
input   fifo_A_out_V_V_full_n;
output   fifo_A_out_V_V_write;
input  [63:0] fifo_B_in_V_V_dout;
input   fifo_B_in_V_V_empty_n;
output   fifo_B_in_V_V_read;
output  [63:0] fifo_B_out_V_V_din;
input   fifo_B_out_V_V_full_n;
output   fifo_B_out_V_V_write;
output  [31:0] fifo_C_drain_out_V_din;
input   fifo_C_drain_out_V_full_n;
output   fifo_C_drain_out_V_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg fifo_A_in_V_V_read;
reg fifo_A_out_V_V_write;
reg fifo_B_in_V_V_read;
reg fifo_B_out_V_V_write;
reg fifo_C_drain_out_V_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [5:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    fifo_A_in_V_V_blk_n;
wire    ap_CS_fsm_pp1_stage0;
reg    ap_enable_reg_pp1_iter1;
wire    ap_block_pp1_stage0;
reg   [0:0] icmp_ln449_reg_758;
reg    fifo_A_out_V_V_blk_n;
reg    fifo_B_in_V_V_blk_n;
reg    fifo_B_out_V_V_blk_n;
reg    fifo_C_drain_out_V_blk_n;
reg    ap_enable_reg_pp1_iter3;
reg   [0:0] select_ln480_12_reg_772;
reg   [0:0] select_ln480_12_reg_772_pp1_iter2_reg;
reg   [4:0] indvar_flatten46_reg_254;
reg   [1:0] c2_0_reg_265;
reg   [4:0] indvar_flatten20_reg_276;
reg   [1:0] c5_0_reg_287;
reg   [3:0] indvar_flatten6_reg_298;
reg   [1:0] c61_0_reg_309;
reg   [1:0] c72_0_reg_320;
wire   [0:0] icmp_ln435_fu_331_p2;
wire    ap_CS_fsm_state2;
wire   [2:0] add_ln435_fu_337_p2;
reg   [2:0] add_ln435_reg_735;
wire   [2:0] add_ln440_fu_349_p2;
wire    ap_CS_fsm_state3;
wire   [1:0] select_ln446_4_fu_375_p3;
wire   [0:0] icmp_ln440_fu_343_p2;
wire   [1:0] c7_fu_410_p2;
wire   [0:0] icmp_ln449_fu_434_p2;
wire    ap_block_state5_pp1_stage0_iter0;
reg    ap_block_state6_pp1_stage0_iter1;
wire    ap_block_state7_pp1_stage0_iter2;
reg    ap_block_state8_pp1_stage0_iter3;
reg    ap_block_pp1_stage0_11001;
reg   [0:0] icmp_ln449_reg_758_pp1_iter1_reg;
reg   [0:0] icmp_ln449_reg_758_pp1_iter2_reg;
wire   [4:0] add_ln449_fu_440_p2;
reg    ap_enable_reg_pp1_iter0;
wire   [1:0] select_ln449_fu_516_p3;
wire   [0:0] select_ln480_12_fu_556_p3;
reg   [0:0] select_ln480_12_reg_772_pp1_iter1_reg;
wire   [1:0] select_ln452_fu_582_p3;
wire   [1:0] select_ln479_fu_608_p3;
reg   [1:0] select_ln479_reg_781;
wire   [1:0] select_ln479_4_fu_616_p3;
reg   [1:0] select_ln479_4_reg_786;
wire   [1:0] c7_4_fu_624_p2;
wire   [3:0] select_ln454_fu_636_p3;
wire   [4:0] select_ln452_4_fu_650_p3;
wire  signed [31:0] p_Repl2_11_fu_661_p1;
reg  signed [31:0] p_Repl2_11_reg_807;
reg  signed [31:0] p_Repl2_12_reg_812;
wire  signed [31:0] p_Repl2_s_fu_675_p1;
reg  signed [31:0] p_Repl2_s_reg_817;
reg  signed [31:0] p_Repl2_10_reg_822;
reg   [1:0] local_C_addr_4_reg_827;
reg   [1:0] local_C_addr_4_reg_827_pp1_iter2_reg;
wire   [31:0] mul_ln479_fu_711_p2;
reg   [31:0] mul_ln479_reg_833;
wire   [31:0] mul_ln479_1_fu_715_p2;
reg   [31:0] mul_ln479_1_reg_838;
wire    ap_CS_fsm_state4;
reg    ap_block_pp1_stage0_subdone;
reg    ap_condition_pp1_exit_iter0_state5;
reg    ap_enable_reg_pp1_iter2;
reg   [1:0] local_C_address0;
reg    local_C_ce0;
reg    local_C_we0;
wire   [31:0] local_C_q0;
reg    local_C_ce1;
reg    local_C_we1;
reg   [2:0] indvar_flatten53_reg_210;
reg    ap_block_state1;
wire    ap_CS_fsm_state9;
reg   [2:0] indvar_flatten_reg_221;
reg   [1:0] c6_0_reg_232;
reg   [1:0] c7_0_reg_243;
reg   [1:0] ap_phi_mux_c61_0_phi_fu_313_p4;
wire   [63:0] zext_ln446_4_fu_405_p1;
wire   [63:0] zext_ln479_8_fu_706_p1;
reg    ap_block_pp1_stage0_01001;
wire   [31:0] tmp_5_fu_723_p2;
wire   [0:0] icmp_ln442_fu_361_p2;
wire   [1:0] c6_fu_355_p2;
wire   [1:0] select_ln446_fu_367_p3;
wire   [2:0] tmp_2_fu_387_p3;
wire   [3:0] zext_ln442_fu_383_p1;
wire   [3:0] zext_ln446_fu_395_p1;
wire   [3:0] add_ln446_fu_399_p2;
wire   [0:0] icmp_ln480_fu_416_p2;
wire   [0:0] icmp_ln480_1_fu_422_p2;
wire   [0:0] icmp_ln452_fu_452_p2;
wire   [0:0] icmp_ln480_8_fu_466_p2;
wire   [0:0] and_ln480_fu_428_p2;
wire   [0:0] xor_ln480_fu_480_p2;
wire   [0:0] icmp_ln456_fu_492_p2;
wire   [0:0] icmp_ln454_fu_504_p2;
wire   [1:0] c2_fu_446_p2;
wire   [1:0] select_ln480_fu_458_p3;
wire   [0:0] and_ln480_18_fu_510_p2;
wire   [0:0] or_ln480_fu_530_p2;
wire   [0:0] select_ln480_10_fu_472_p3;
wire   [0:0] icmp_ln480_9_fu_544_p2;
wire   [0:0] and_ln480_19_fu_550_p2;
wire   [0:0] and_ln480_16_fu_486_p2;
wire   [0:0] xor_ln480_4_fu_564_p2;
wire   [0:0] and_ln480_17_fu_498_p2;
wire   [0:0] or_ln480_4_fu_570_p2;
wire   [1:0] c5_fu_524_p2;
wire   [1:0] select_ln480_11_fu_536_p3;
wire   [0:0] and_ln480_20_fu_576_p2;
wire   [0:0] or_ln479_fu_596_p2;
wire   [0:0] or_ln479_4_fu_602_p2;
wire   [1:0] c6_6_fu_590_p2;
wire   [3:0] add_ln454_4_fu_630_p2;
wire   [4:0] add_ln452_4_fu_644_p2;
wire   [2:0] tmp_3_fu_689_p3;
wire   [3:0] zext_ln479_7_fu_696_p1;
wire   [3:0] zext_ln479_fu_658_p1;
wire   [3:0] add_ln479_7_fu_700_p2;
wire   [31:0] add_ln479_fu_719_p2;
reg   [5:0] ap_NS_fsm;
reg    ap_idle_pp1;
wire    ap_enable_pp1;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 6'd1;
#0 ap_enable_reg_pp1_iter1 = 1'b0;
#0 ap_enable_reg_pp1_iter3 = 1'b0;
#0 ap_enable_reg_pp1_iter0 = 1'b0;
#0 ap_enable_reg_pp1_iter2 = 1'b0;
end

PE139_local_C #(
    .DataWidth( 32 ),
    .AddressRange( 4 ),
    .AddressWidth( 2 ))
local_C_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(local_C_address0),
    .ce0(local_C_ce0),
    .we0(local_C_we0),
    .d0(32'd0),
    .q0(local_C_q0),
    .address1(local_C_addr_4_reg_827_pp1_iter2_reg),
    .ce1(local_C_ce1),
    .we1(local_C_we1),
    .d1(tmp_5_fu_723_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((icmp_ln435_fu_331_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp1_iter0 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp1_stage0_subdone) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b1 == ap_condition_pp1_exit_iter0_state5))) begin
            ap_enable_reg_pp1_iter0 <= 1'b0;
        end else if ((1'b1 == ap_CS_fsm_state4)) begin
            ap_enable_reg_pp1_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp1_iter1 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp1_stage0_subdone)) begin
            if ((1'b1 == ap_condition_pp1_exit_iter0_state5)) begin
                ap_enable_reg_pp1_iter1 <= (1'b1 ^ ap_condition_pp1_exit_iter0_state5);
            end else if ((1'b1 == 1'b1)) begin
                ap_enable_reg_pp1_iter1 <= ap_enable_reg_pp1_iter0;
            end
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp1_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp1_stage0_subdone)) begin
            ap_enable_reg_pp1_iter2 <= ap_enable_reg_pp1_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp1_iter3 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp1_stage0_subdone)) begin
            ap_enable_reg_pp1_iter3 <= ap_enable_reg_pp1_iter2;
        end else if ((1'b1 == ap_CS_fsm_state4)) begin
            ap_enable_reg_pp1_iter3 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((internal_ap_ready == 1'b0) & (real_start == 1'b1))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        c2_0_reg_265 <= 2'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        c2_0_reg_265 <= select_ln449_fu_516_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        c5_0_reg_287 <= 2'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        c5_0_reg_287 <= select_ln452_fu_582_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        c61_0_reg_309 <= 2'd0;
    end else if (((icmp_ln449_reg_758 == 1'd0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        c61_0_reg_309 <= select_ln479_4_reg_786;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln440_fu_343_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        c6_0_reg_232 <= select_ln446_4_fu_375_p3;
    end else if (((icmp_ln435_fu_331_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        c6_0_reg_232 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        c72_0_reg_320 <= 2'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        c72_0_reg_320 <= c7_4_fu_624_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln440_fu_343_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        c7_0_reg_243 <= c7_fu_410_p2;
    end else if (((icmp_ln435_fu_331_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        c7_0_reg_243 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        indvar_flatten20_reg_276 <= 5'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        indvar_flatten20_reg_276 <= select_ln452_4_fu_650_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        indvar_flatten46_reg_254 <= 5'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        indvar_flatten46_reg_254 <= add_ln449_fu_440_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        indvar_flatten53_reg_210 <= add_ln435_reg_735;
    end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        indvar_flatten53_reg_210 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        indvar_flatten6_reg_298 <= 4'd0;
    end else if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        indvar_flatten6_reg_298 <= select_ln454_fu_636_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln440_fu_343_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        indvar_flatten_reg_221 <= add_ln440_fu_349_p2;
    end else if (((icmp_ln435_fu_331_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        indvar_flatten_reg_221 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        add_ln435_reg_735 <= add_ln435_fu_337_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        icmp_ln449_reg_758 <= icmp_ln449_fu_434_p2;
        icmp_ln449_reg_758_pp1_iter1_reg <= icmp_ln449_reg_758;
        select_ln480_12_reg_772_pp1_iter1_reg <= select_ln480_12_reg_772;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp1_stage0_11001)) begin
        icmp_ln449_reg_758_pp1_iter2_reg <= icmp_ln449_reg_758_pp1_iter1_reg;
        local_C_addr_4_reg_827_pp1_iter2_reg <= local_C_addr_4_reg_827;
        select_ln480_12_reg_772_pp1_iter2_reg <= select_ln480_12_reg_772_pp1_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        local_C_addr_4_reg_827 <= zext_ln479_8_fu_706_p1;
        p_Repl2_10_reg_822 <= {{fifo_B_in_V_V_dout[63:32]}};
        p_Repl2_11_reg_807 <= p_Repl2_11_fu_661_p1;
        p_Repl2_12_reg_812 <= {{fifo_A_in_V_V_dout[63:32]}};
        p_Repl2_s_reg_817 <= p_Repl2_s_fu_675_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln449_reg_758_pp1_iter1_reg == 1'd0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        mul_ln479_1_reg_838 <= mul_ln479_1_fu_715_p2;
        mul_ln479_reg_833 <= mul_ln479_fu_711_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (ap_enable_reg_pp1_iter0 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        select_ln479_4_reg_786 <= select_ln479_4_fu_616_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln449_fu_434_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        select_ln479_reg_781 <= select_ln479_fu_608_p3;
        select_ln480_12_reg_772 <= select_ln480_12_fu_556_p3;
    end
end

always @ (*) begin
    if ((icmp_ln449_fu_434_p2 == 1'd1)) begin
        ap_condition_pp1_exit_iter0_state5 = 1'b1;
    end else begin
        ap_condition_pp1_exit_iter0_state5 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln435_fu_331_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((real_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp1_iter3 == 1'b0) & (ap_enable_reg_pp1_iter1 == 1'b0) & (ap_enable_reg_pp1_iter2 == 1'b0) & (ap_enable_reg_pp1_iter0 == 1'b0))) begin
        ap_idle_pp1 = 1'b1;
    end else begin
        ap_idle_pp1 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0))) begin
        ap_phi_mux_c61_0_phi_fu_313_p4 = select_ln479_4_reg_786;
    end else begin
        ap_phi_mux_c61_0_phi_fu_313_p4 = c61_0_reg_309;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0))) begin
        fifo_A_in_V_V_blk_n = fifo_A_in_V_V_empty_n;
    end else begin
        fifo_A_in_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        fifo_A_in_V_V_read = 1'b1;
    end else begin
        fifo_A_in_V_V_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0))) begin
        fifo_A_out_V_V_blk_n = fifo_A_out_V_V_full_n;
    end else begin
        fifo_A_out_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        fifo_A_out_V_V_write = 1'b1;
    end else begin
        fifo_A_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0))) begin
        fifo_B_in_V_V_blk_n = fifo_B_in_V_V_empty_n;
    end else begin
        fifo_B_in_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        fifo_B_in_V_V_read = 1'b1;
    end else begin
        fifo_B_in_V_V_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0))) begin
        fifo_B_out_V_V_blk_n = fifo_B_out_V_V_full_n;
    end else begin
        fifo_B_out_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758 == 1'd0) & (ap_enable_reg_pp1_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp1_stage0) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        fifo_B_out_V_V_write = 1'b1;
    end else begin
        fifo_B_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter3 == 1'b1))) begin
        fifo_C_drain_out_V_blk_n = fifo_C_drain_out_V_full_n;
    end else begin
        fifo_C_drain_out_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (ap_enable_reg_pp1_iter3 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        fifo_C_drain_out_V_write = 1'b1;
    end else begin
        fifo_C_drain_out_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln435_fu_331_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp1_stage0) & (ap_enable_reg_pp1_iter2 == 1'b1))) begin
        local_C_address0 = local_C_addr_4_reg_827;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        local_C_address0 = zext_ln446_4_fu_405_p1;
    end else begin
        local_C_address0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((ap_enable_reg_pp1_iter2 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001)))) begin
        local_C_ce0 = 1'b1;
    end else begin
        local_C_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp1_iter3 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        local_C_ce1 = 1'b1;
    end else begin
        local_C_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln440_fu_343_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        local_C_we0 = 1'b1;
    end else begin
        local_C_we0 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln449_reg_758_pp1_iter2_reg == 1'd0) & (ap_enable_reg_pp1_iter3 == 1'b1) & (1'b0 == ap_block_pp1_stage0_11001))) begin
        local_C_we1 = 1'b1;
    end else begin
        local_C_we1 = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (start_full_n == 1'b0))) begin
        real_start = 1'b0;
    end else begin
        real_start = ap_start;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (real_start == 1'b1))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((icmp_ln435_fu_331_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((icmp_ln440_fu_343_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_pp1_stage0;
        end
        ap_ST_fsm_pp1_stage0 : begin
            if ((~((icmp_ln449_fu_434_p2 == 1'd1) & (ap_enable_reg_pp1_iter1 == 1'b0) & (1'b0 == ap_block_pp1_stage0_subdone) & (ap_enable_reg_pp1_iter0 == 1'b1)) & ~((ap_enable_reg_pp1_iter2 == 1'b0) & (1'b0 == ap_block_pp1_stage0_subdone) & (ap_enable_reg_pp1_iter3 == 1'b1)))) begin
                ap_NS_fsm = ap_ST_fsm_pp1_stage0;
            end else if ((((ap_enable_reg_pp1_iter2 == 1'b0) & (1'b0 == ap_block_pp1_stage0_subdone) & (ap_enable_reg_pp1_iter3 == 1'b1)) | ((icmp_ln449_fu_434_p2 == 1'd1) & (ap_enable_reg_pp1_iter1 == 1'b0) & (1'b0 == ap_block_pp1_stage0_subdone) & (ap_enable_reg_pp1_iter0 == 1'b1)))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp1_stage0;
            end
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state2;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln435_fu_337_p2 = (indvar_flatten53_reg_210 + 3'd1);

assign add_ln440_fu_349_p2 = (indvar_flatten_reg_221 + 3'd1);

assign add_ln446_fu_399_p2 = (zext_ln442_fu_383_p1 + zext_ln446_fu_395_p1);

assign add_ln449_fu_440_p2 = (indvar_flatten46_reg_254 + 5'd1);

assign add_ln452_4_fu_644_p2 = (indvar_flatten20_reg_276 + 5'd1);

assign add_ln454_4_fu_630_p2 = (indvar_flatten6_reg_298 + 4'd1);

assign add_ln479_7_fu_700_p2 = (zext_ln479_7_fu_696_p1 + zext_ln479_fu_658_p1);

assign add_ln479_fu_719_p2 = (mul_ln479_1_reg_838 + mul_ln479_reg_833);

assign and_ln480_16_fu_486_p2 = (xor_ln480_fu_480_p2 & and_ln480_fu_428_p2);

assign and_ln480_17_fu_498_p2 = (xor_ln480_fu_480_p2 & icmp_ln456_fu_492_p2);

assign and_ln480_18_fu_510_p2 = (xor_ln480_fu_480_p2 & icmp_ln454_fu_504_p2);

assign and_ln480_19_fu_550_p2 = (select_ln480_10_fu_472_p3 & icmp_ln480_9_fu_544_p2);

assign and_ln480_20_fu_576_p2 = (or_ln480_4_fu_570_p2 & and_ln480_17_fu_498_p2);

assign and_ln480_fu_428_p2 = (icmp_ln480_fu_416_p2 & icmp_ln480_1_fu_422_p2);

assign ap_CS_fsm_pp1_stage0 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd5];

assign ap_block_pp1_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp1_stage0_01001 = (((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (fifo_C_drain_out_V_full_n == 1'b0) & (ap_enable_reg_pp1_iter3 == 1'b1)) | ((ap_enable_reg_pp1_iter1 == 1'b1) & (((icmp_ln449_reg_758 == 1'd0) & (fifo_A_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_A_in_V_V_empty_n == 1'b0)))));
end

always @ (*) begin
    ap_block_pp1_stage0_11001 = (((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (fifo_C_drain_out_V_full_n == 1'b0) & (ap_enable_reg_pp1_iter3 == 1'b1)) | ((ap_enable_reg_pp1_iter1 == 1'b1) & (((icmp_ln449_reg_758 == 1'd0) & (fifo_A_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_A_in_V_V_empty_n == 1'b0)))));
end

always @ (*) begin
    ap_block_pp1_stage0_subdone = (((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (fifo_C_drain_out_V_full_n == 1'b0) & (ap_enable_reg_pp1_iter3 == 1'b1)) | ((ap_enable_reg_pp1_iter1 == 1'b1) & (((icmp_ln449_reg_758 == 1'd0) & (fifo_A_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_A_in_V_V_empty_n == 1'b0)))));
end

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

assign ap_block_state5_pp1_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state6_pp1_stage0_iter1 = (((icmp_ln449_reg_758 == 1'd0) & (fifo_A_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0)) | ((icmp_ln449_reg_758 == 1'd0) & (fifo_A_in_V_V_empty_n == 1'b0)));
end

assign ap_block_state7_pp1_stage0_iter2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state8_pp1_stage0_iter3 = ((select_ln480_12_reg_772_pp1_iter2_reg == 1'd1) & (fifo_C_drain_out_V_full_n == 1'b0));
end

assign ap_enable_pp1 = (ap_idle_pp1 ^ 1'b1);

assign ap_ready = internal_ap_ready;

assign c2_fu_446_p2 = (2'd1 + c2_0_reg_265);

assign c5_fu_524_p2 = (2'd1 + select_ln480_fu_458_p3);

assign c6_6_fu_590_p2 = (2'd1 + select_ln480_11_fu_536_p3);

assign c6_fu_355_p2 = (c6_0_reg_232 + 2'd1);

assign c7_4_fu_624_p2 = (select_ln479_fu_608_p3 + 2'd1);

assign c7_fu_410_p2 = (select_ln446_fu_367_p3 + 2'd1);

assign fifo_A_out_V_V_din = fifo_A_in_V_V_dout;

assign fifo_B_out_V_V_din = fifo_B_in_V_V_dout;

assign fifo_C_drain_out_V_din = tmp_5_fu_723_p2;

assign icmp_ln435_fu_331_p2 = ((indvar_flatten53_reg_210 == 3'd4) ? 1'b1 : 1'b0);

assign icmp_ln440_fu_343_p2 = ((indvar_flatten_reg_221 == 3'd4) ? 1'b1 : 1'b0);

assign icmp_ln442_fu_361_p2 = ((c7_0_reg_243 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln449_fu_434_p2 = ((indvar_flatten46_reg_254 == 5'd16) ? 1'b1 : 1'b0);

assign icmp_ln452_fu_452_p2 = ((indvar_flatten20_reg_276 == 5'd8) ? 1'b1 : 1'b0);

assign icmp_ln454_fu_504_p2 = ((indvar_flatten6_reg_298 == 4'd4) ? 1'b1 : 1'b0);

assign icmp_ln456_fu_492_p2 = ((c72_0_reg_320 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln480_1_fu_422_p2 = ((c5_0_reg_287 == 2'd1) ? 1'b1 : 1'b0);

assign icmp_ln480_8_fu_466_p2 = ((c2_0_reg_265 == 2'd0) ? 1'b1 : 1'b0);

assign icmp_ln480_9_fu_544_p2 = ((select_ln480_fu_458_p3 == 2'd0) ? 1'b1 : 1'b0);

assign icmp_ln480_fu_416_p2 = ((c2_0_reg_265 == 2'd1) ? 1'b1 : 1'b0);

assign mul_ln479_1_fu_715_p2 = ($signed(p_Repl2_12_reg_812) * $signed(p_Repl2_10_reg_822));

assign mul_ln479_fu_711_p2 = ($signed(p_Repl2_11_reg_807) * $signed(p_Repl2_s_reg_817));

assign or_ln479_4_fu_602_p2 = (or_ln479_fu_596_p2 | icmp_ln452_fu_452_p2);

assign or_ln479_fu_596_p2 = (and_ln480_20_fu_576_p2 | and_ln480_18_fu_510_p2);

assign or_ln480_4_fu_570_p2 = (xor_ln480_4_fu_564_p2 | icmp_ln452_fu_452_p2);

assign or_ln480_fu_530_p2 = (icmp_ln452_fu_452_p2 | and_ln480_18_fu_510_p2);

assign p_Repl2_11_fu_661_p1 = fifo_A_in_V_V_dout[31:0];

assign p_Repl2_s_fu_675_p1 = fifo_B_in_V_V_dout[31:0];

assign select_ln446_4_fu_375_p3 = ((icmp_ln442_fu_361_p2[0:0] === 1'b1) ? c6_fu_355_p2 : c6_0_reg_232);

assign select_ln446_fu_367_p3 = ((icmp_ln442_fu_361_p2[0:0] === 1'b1) ? 2'd0 : c7_0_reg_243);

assign select_ln449_fu_516_p3 = ((icmp_ln452_fu_452_p2[0:0] === 1'b1) ? c2_fu_446_p2 : c2_0_reg_265);

assign select_ln452_4_fu_650_p3 = ((icmp_ln452_fu_452_p2[0:0] === 1'b1) ? 5'd1 : add_ln452_4_fu_644_p2);

assign select_ln452_fu_582_p3 = ((and_ln480_18_fu_510_p2[0:0] === 1'b1) ? c5_fu_524_p2 : select_ln480_fu_458_p3);

assign select_ln454_fu_636_p3 = ((or_ln480_fu_530_p2[0:0] === 1'b1) ? 4'd1 : add_ln454_4_fu_630_p2);

assign select_ln479_4_fu_616_p3 = ((and_ln480_20_fu_576_p2[0:0] === 1'b1) ? c6_6_fu_590_p2 : select_ln480_11_fu_536_p3);

assign select_ln479_fu_608_p3 = ((or_ln479_4_fu_602_p2[0:0] === 1'b1) ? 2'd0 : c72_0_reg_320);

assign select_ln480_10_fu_472_p3 = ((icmp_ln452_fu_452_p2[0:0] === 1'b1) ? icmp_ln480_8_fu_466_p2 : icmp_ln480_fu_416_p2);

assign select_ln480_11_fu_536_p3 = ((or_ln480_fu_530_p2[0:0] === 1'b1) ? 2'd0 : ap_phi_mux_c61_0_phi_fu_313_p4);

assign select_ln480_12_fu_556_p3 = ((and_ln480_18_fu_510_p2[0:0] === 1'b1) ? and_ln480_19_fu_550_p2 : and_ln480_16_fu_486_p2);

assign select_ln480_fu_458_p3 = ((icmp_ln452_fu_452_p2[0:0] === 1'b1) ? 2'd0 : c5_0_reg_287);

assign start_out = real_start;

assign tmp_2_fu_387_p3 = {{select_ln446_fu_367_p3}, {1'd0}};

assign tmp_3_fu_689_p3 = {{select_ln479_reg_781}, {1'd0}};

assign tmp_5_fu_723_p2 = (local_C_q0 + add_ln479_fu_719_p2);

assign xor_ln480_4_fu_564_p2 = (icmp_ln454_fu_504_p2 ^ 1'd1);

assign xor_ln480_fu_480_p2 = (icmp_ln452_fu_452_p2 ^ 1'd1);

assign zext_ln442_fu_383_p1 = select_ln446_4_fu_375_p3;

assign zext_ln446_4_fu_405_p1 = add_ln446_fu_399_p2;

assign zext_ln446_fu_395_p1 = tmp_2_fu_387_p3;

assign zext_ln479_7_fu_696_p1 = tmp_3_fu_689_p3;

assign zext_ln479_8_fu_706_p1 = add_ln479_7_fu_700_p2;

assign zext_ln479_fu_658_p1 = select_ln479_4_reg_786;

endmodule //PE
