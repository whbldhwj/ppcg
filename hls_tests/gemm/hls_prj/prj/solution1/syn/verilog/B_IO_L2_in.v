// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module B_IO_L2_in (
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
        fifo_B_in_V_V_dout,
        fifo_B_in_V_V_empty_n,
        fifo_B_in_V_V_read,
        fifo_B_out_V_V_din,
        fifo_B_out_V_V_full_n,
        fifo_B_out_V_V_write,
        fifo_B_local_out_V_V_din,
        fifo_B_local_out_V_V_full_n,
        fifo_B_local_out_V_V_write
);

parameter    ap_ST_fsm_state1 = 6'd1;
parameter    ap_ST_fsm_state2 = 6'd2;
parameter    ap_ST_fsm_state3 = 6'd4;
parameter    ap_ST_fsm_state4 = 6'd8;
parameter    ap_ST_fsm_state5 = 6'd16;
parameter    ap_ST_fsm_state6 = 6'd32;

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
input  [127:0] fifo_B_in_V_V_dout;
input   fifo_B_in_V_V_empty_n;
output   fifo_B_in_V_V_read;
output  [127:0] fifo_B_out_V_V_din;
input   fifo_B_out_V_V_full_n;
output   fifo_B_out_V_V_write;
output  [63:0] fifo_B_local_out_V_V_din;
input   fifo_B_local_out_V_V_full_n;
output   fifo_B_local_out_V_V_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg fifo_B_in_V_V_read;
reg fifo_B_out_V_V_write;
reg fifo_B_local_out_V_V_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [5:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
wire   [1:0] c0_fu_163_p2;
reg   [1:0] c0_reg_214;
wire    ap_CS_fsm_state2;
reg   [0:0] intra_trans_en_0_loa_2_reg_219;
wire   [0:0] icmp_ln343_fu_157_p2;
wire   [1:0] c1_fu_175_p2;
reg   [1:0] c1_reg_227;
wire    ap_CS_fsm_state3;
wire   [1:0] c2_fu_187_p2;
reg   [1:0] c2_reg_235;
wire    ap_CS_fsm_state4;
reg   [0:0] intra_trans_en_0_loa_reg_240;
wire   [0:0] icmp_ln345_fu_181_p2;
wire   [0:0] xor_ln362_fu_193_p2;
wire    ap_CS_fsm_state5;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_ap_ready;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_ap_done;
reg   [0:0] arb_2_reg_108;
wire    grp_B_IO_L2_in_intra_tra_fu_131_ap_ready;
wire    grp_B_IO_L2_in_intra_tra_fu_131_ap_done;
reg    ap_block_state5_on_subcall_done;
reg   [0:0] local_B_ping_0_V_address0;
reg    local_B_ping_0_V_ce0;
reg    local_B_ping_0_V_we0;
wire   [127:0] local_B_ping_0_V_q0;
reg   [0:0] local_B_pong_0_V_address0;
reg    local_B_pong_0_V_ce0;
reg    local_B_pong_0_V_we0;
wire   [127:0] local_B_pong_0_V_q0;
wire    grp_B_IO_L2_in_intra_tra_fu_131_ap_start;
wire    grp_B_IO_L2_in_intra_tra_fu_131_ap_idle;
wire   [0:0] grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_address0;
wire    grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_ce0;
reg   [127:0] grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_q0;
wire   [63:0] grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_din;
wire    grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_write;
reg    grp_B_IO_L2_in_intra_tra_fu_131_en;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_ap_idle;
wire   [0:0] grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_address0;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_ce0;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_we0;
wire   [127:0] grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_d0;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_in_V_V_read;
wire   [127:0] grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_din;
wire    grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_write;
reg   [1:0] c0_prev_reg_86;
reg    ap_block_state1;
wire   [0:0] icmp_ln344_fu_169_p2;
reg   [1:0] c1_prev_reg_97;
wire   [0:0] ap_phi_mux_arb_2_phi_fu_112_p4;
reg   [1:0] c2_prev_reg_120;
reg    grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg;
wire    ap_CS_fsm_state6;
reg    grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg;
reg   [0:0] intra_trans_en_0_fu_74;
reg   [5:0] ap_NS_fsm;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 6'd1;
#0 grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg = 1'b0;
#0 grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg = 1'b0;
end

B_IO_L2_in_local_fYi #(
    .DataWidth( 128 ),
    .AddressRange( 2 ),
    .AddressWidth( 1 ))
local_B_ping_0_V_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(local_B_ping_0_V_address0),
    .ce0(local_B_ping_0_V_ce0),
    .we0(local_B_ping_0_V_we0),
    .d0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_d0),
    .q0(local_B_ping_0_V_q0)
);

B_IO_L2_in_local_fYi #(
    .DataWidth( 128 ),
    .AddressRange( 2 ),
    .AddressWidth( 1 ))
local_B_pong_0_V_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(local_B_pong_0_V_address0),
    .ce0(local_B_pong_0_V_ce0),
    .we0(local_B_pong_0_V_we0),
    .d0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_d0),
    .q0(local_B_pong_0_V_q0)
);

B_IO_L2_in_intra_tra grp_B_IO_L2_in_intra_tra_fu_131(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_B_IO_L2_in_intra_tra_fu_131_ap_start),
    .ap_done(grp_B_IO_L2_in_intra_tra_fu_131_ap_done),
    .ap_idle(grp_B_IO_L2_in_intra_tra_fu_131_ap_idle),
    .ap_ready(grp_B_IO_L2_in_intra_tra_fu_131_ap_ready),
    .local_B_0_V_address0(grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_address0),
    .local_B_0_V_ce0(grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_ce0),
    .local_B_0_V_q0(grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_q0),
    .fifo_B_local_out_V_V_din(grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_din),
    .fifo_B_local_out_V_V_full_n(fifo_B_local_out_V_V_full_n),
    .fifo_B_local_out_V_V_write(grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_write),
    .en(grp_B_IO_L2_in_intra_tra_fu_131_en)
);

B_IO_L2_in_inter_tra_1 grp_B_IO_L2_in_inter_tra_1_fu_139(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start),
    .ap_done(grp_B_IO_L2_in_inter_tra_1_fu_139_ap_done),
    .ap_idle(grp_B_IO_L2_in_inter_tra_1_fu_139_ap_idle),
    .ap_ready(grp_B_IO_L2_in_inter_tra_1_fu_139_ap_ready),
    .local_B_0_V_address0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_address0),
    .local_B_0_V_ce0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_ce0),
    .local_B_0_V_we0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_we0),
    .local_B_0_V_d0(grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_d0),
    .fifo_B_in_V_V_dout(fifo_B_in_V_V_dout),
    .fifo_B_in_V_V_empty_n(fifo_B_in_V_V_empty_n),
    .fifo_B_in_V_V_read(grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_in_V_V_read),
    .fifo_B_out_V_V_din(grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_din),
    .fifo_B_out_V_V_full_n(fifo_B_out_V_V_full_n),
    .fifo_B_out_V_V_write(grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_write)
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
        end else if (((1'b1 == ap_CS_fsm_state6) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg <= 1'b0;
    end else begin
        if ((((icmp_ln345_fu_181_p2 == 1'd0) & (ap_phi_mux_arb_2_phi_fu_112_p4 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((ap_phi_mux_arb_2_phi_fu_112_p4 == 1'd0) & (icmp_ln345_fu_181_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
            grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg <= 1'b1;
        end else if ((grp_B_IO_L2_in_inter_tra_1_fu_139_ap_ready == 1'b1)) begin
            grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg <= 1'b0;
    end else begin
        if ((((icmp_ln345_fu_181_p2 == 1'd0) & (ap_phi_mux_arb_2_phi_fu_112_p4 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((ap_phi_mux_arb_2_phi_fu_112_p4 == 1'd0) & (icmp_ln345_fu_181_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)) | ((icmp_ln343_fu_157_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)))) begin
            grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg <= 1'b1;
        end else if ((grp_B_IO_L2_in_intra_tra_fu_131_ap_ready == 1'b1)) begin
            grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg <= 1'b0;
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
    if (((icmp_ln344_fu_169_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        arb_2_reg_108 <= 1'd0;
    end else if (((1'b1 == ap_CS_fsm_state5) & (1'b0 == ap_block_state5_on_subcall_done))) begin
        arb_2_reg_108 <= xor_ln362_fu_193_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln344_fu_169_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        c0_prev_reg_86 <= c0_reg_214;
    end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        c0_prev_reg_86 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln343_fu_157_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        c1_prev_reg_97 <= 2'd0;
    end else if (((icmp_ln345_fu_181_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        c1_prev_reg_97 <= c1_reg_227;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln344_fu_169_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        c2_prev_reg_120 <= 2'd0;
    end else if (((1'b1 == ap_CS_fsm_state5) & (1'b0 == ap_block_state5_on_subcall_done))) begin
        c2_prev_reg_120 <= c2_reg_235;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state5) & (1'b0 == ap_block_state5_on_subcall_done))) begin
        intra_trans_en_0_fu_74 <= 1'd1;
    end else if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        intra_trans_en_0_fu_74 <= 1'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        c0_reg_214 <= c0_fu_163_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        c1_reg_227 <= c1_fu_175_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        c2_reg_235 <= c2_fu_187_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln343_fu_157_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        intra_trans_en_0_loa_2_reg_219 <= intra_trans_en_0_fu_74;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln345_fu_181_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        intra_trans_en_0_loa_reg_240 <= intra_trans_en_0_fu_74;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b1))) begin
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
    if ((((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)) | ((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5)))) begin
        fifo_B_in_V_V_read = grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_in_V_V_read;
    end else begin
        fifo_B_in_V_V_read = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) | ((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)) | ((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5)))) begin
        fifo_B_local_out_V_V_write = grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_write;
    end else begin
        fifo_B_local_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if ((((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)) | ((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5)))) begin
        fifo_B_out_V_V_write = grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_write;
    end else begin
        fifo_B_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if ((((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)) | ((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5)))) begin
        grp_B_IO_L2_in_intra_tra_fu_131_en = intra_trans_en_0_loa_reg_240;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        grp_B_IO_L2_in_intra_tra_fu_131_en = intra_trans_en_0_loa_2_reg_219;
    end else begin
        grp_B_IO_L2_in_intra_tra_fu_131_en = 'bx;
    end
end

always @ (*) begin
    if (((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5))) begin
        grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_q0 = local_B_pong_0_V_q0;
    end else if (((1'b1 == ap_CS_fsm_state6) | ((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)))) begin
        grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_q0 = local_B_ping_0_V_q0;
    end else begin
        grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_q0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b1))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5))) begin
        local_B_ping_0_V_address0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_address0;
    end else if (((1'b1 == ap_CS_fsm_state6) | ((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)))) begin
        local_B_ping_0_V_address0 = grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_address0;
    end else begin
        local_B_ping_0_V_address0 = 'bx;
    end
end

always @ (*) begin
    if (((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5))) begin
        local_B_ping_0_V_ce0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_ce0;
    end else if (((1'b1 == ap_CS_fsm_state6) | ((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5)))) begin
        local_B_ping_0_V_ce0 = grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_ce0;
    end else begin
        local_B_ping_0_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((arb_2_reg_108 == 1'd1) & (1'b1 == ap_CS_fsm_state5))) begin
        local_B_ping_0_V_we0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_we0;
    end else begin
        local_B_ping_0_V_we0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        if ((arb_2_reg_108 == 1'd0)) begin
            local_B_pong_0_V_address0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_address0;
        end else if ((arb_2_reg_108 == 1'd1)) begin
            local_B_pong_0_V_address0 = grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_address0;
        end else begin
            local_B_pong_0_V_address0 = 'bx;
        end
    end else begin
        local_B_pong_0_V_address0 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        if ((arb_2_reg_108 == 1'd0)) begin
            local_B_pong_0_V_ce0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_ce0;
        end else if ((arb_2_reg_108 == 1'd1)) begin
            local_B_pong_0_V_ce0 = grp_B_IO_L2_in_intra_tra_fu_131_local_B_0_V_ce0;
        end else begin
            local_B_pong_0_V_ce0 = 1'b0;
        end
    end else begin
        local_B_pong_0_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((arb_2_reg_108 == 1'd0) & (1'b1 == ap_CS_fsm_state5))) begin
        local_B_pong_0_V_we0 = grp_B_IO_L2_in_inter_tra_1_fu_139_local_B_0_V_we0;
    end else begin
        local_B_pong_0_V_we0 = 1'b0;
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
            if (((icmp_ln343_fu_157_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((icmp_ln344_fu_169_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((icmp_ln345_fu_181_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_CS_fsm_state5) & (1'b0 == ap_block_state5_on_subcall_done))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state5_on_subcall_done = (((arb_2_reg_108 == 1'd0) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b0)) | ((arb_2_reg_108 == 1'd0) & (grp_B_IO_L2_in_inter_tra_1_fu_139_ap_done == 1'b0)) | ((arb_2_reg_108 == 1'd1) & (grp_B_IO_L2_in_inter_tra_1_fu_139_ap_done == 1'b0)) | ((arb_2_reg_108 == 1'd1) & (grp_B_IO_L2_in_intra_tra_fu_131_ap_done == 1'b0)));
end

assign ap_phi_mux_arb_2_phi_fu_112_p4 = arb_2_reg_108;

assign ap_ready = internal_ap_ready;

assign c0_fu_163_p2 = (c0_prev_reg_86 + 2'd1);

assign c1_fu_175_p2 = (c1_prev_reg_97 + 2'd1);

assign c2_fu_187_p2 = (c2_prev_reg_120 + 2'd1);

assign fifo_B_local_out_V_V_din = grp_B_IO_L2_in_intra_tra_fu_131_fifo_B_local_out_V_V_din;

assign fifo_B_out_V_V_din = grp_B_IO_L2_in_inter_tra_1_fu_139_fifo_B_out_V_V_din;

assign grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start = grp_B_IO_L2_in_inter_tra_1_fu_139_ap_start_reg;

assign grp_B_IO_L2_in_intra_tra_fu_131_ap_start = grp_B_IO_L2_in_intra_tra_fu_131_ap_start_reg;

assign icmp_ln343_fu_157_p2 = ((c0_prev_reg_86 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln344_fu_169_p2 = ((c1_prev_reg_97 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln345_fu_181_p2 = ((c2_prev_reg_120 == 2'd2) ? 1'b1 : 1'b0);

assign start_out = real_start;

assign xor_ln362_fu_193_p2 = (arb_2_reg_108 ^ 1'd1);

endmodule //B_IO_L2_in
