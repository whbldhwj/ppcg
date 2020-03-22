// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module C_drain_IO_L1_out_in_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        local_C_0_V_address0,
        local_C_0_V_ce0,
        local_C_0_V_q0,
        fifo_C_drain_in_V_V_dout,
        fifo_C_drain_in_V_V_empty_n,
        fifo_C_drain_in_V_V_read,
        fifo_C_drain_out_V_V_din,
        fifo_C_drain_out_V_V_full_n,
        fifo_C_drain_out_V_V_write,
        en
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_pp0_stage0 = 3'd2;
parameter    ap_ST_fsm_state4 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [0:0] local_C_0_V_address0;
output   local_C_0_V_ce0;
input  [63:0] local_C_0_V_q0;
input  [63:0] fifo_C_drain_in_V_V_dout;
input   fifo_C_drain_in_V_V_empty_n;
output   fifo_C_drain_in_V_V_read;
output  [63:0] fifo_C_drain_out_V_V_din;
input   fifo_C_drain_out_V_V_full_n;
output   fifo_C_drain_out_V_V_write;
input   en;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg local_C_0_V_ce0;
reg fifo_C_drain_in_V_V_read;
reg fifo_C_drain_out_V_V_write;

(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    fifo_C_drain_in_V_V_blk_n;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
reg   [0:0] icmp_ln610_reg_230;
reg   [0:0] select_ln616_1_reg_239;
reg    fifo_C_drain_out_V_V_blk_n;
reg   [2:0] indvar_flatten_reg_110;
reg   [1:0] c4_0_reg_121;
reg   [1:0] c5_0_reg_132;
wire   [0:0] en_read_read_fu_78_p2;
wire   [0:0] icmp_ln610_fu_155_p2;
wire    ap_block_state2_pp0_stage0_iter0;
reg    ap_predicate_op31_read_state3;
reg    ap_block_state3_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_11001;
wire   [2:0] add_ln610_fu_161_p2;
reg    ap_enable_reg_pp0_iter0;
wire   [0:0] select_ln616_1_fu_199_p3;
wire   [1:0] select_ln610_fu_207_p3;
wire   [1:0] c5_fu_220_p2;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
reg   [63:0] ap_phi_mux_tmp_V_phi_fu_146_p4;
wire   [63:0] ap_phi_reg_pp0_iter1_tmp_V_reg_143;
wire   [63:0] zext_ln617_fu_215_p1;
reg    ap_block_pp0_stage0_01001;
wire   [0:0] icmp_ln612_fu_173_p2;
wire   [0:0] icmp_ln616_fu_187_p2;
wire   [0:0] icmp_ln616_1_fu_193_p2;
wire   [1:0] c4_fu_167_p2;
wire   [1:0] select_ln616_fu_179_p3;
wire    ap_CS_fsm_state4;
reg   [2:0] ap_NS_fsm;
reg    ap_idle_pp0;
wire    ap_enable_pp0;

// power-on initialization
initial begin
#0 ap_CS_fsm = 3'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter0 = 1'b0;
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b1 == ap_condition_pp0_exit_iter0_state2) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter0 <= 1'b0;
        end else if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b1 == ap_condition_pp0_exit_iter0_state2) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter1 <= (1'b1 ^ ap_condition_pp0_exit_iter0_state2);
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end else if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        c4_0_reg_121 <= 2'd0;
    end else if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln610_fu_155_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        c4_0_reg_121 <= select_ln610_fu_207_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        c5_0_reg_132 <= 2'd0;
    end else if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln610_fu_155_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        c5_0_reg_132 <= c5_fu_220_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        indvar_flatten_reg_110 <= 3'd0;
    end else if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln610_fu_155_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        indvar_flatten_reg_110 <= add_ln610_fu_161_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        icmp_ln610_reg_230 <= icmp_ln610_fu_155_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln610_fu_155_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        select_ln616_1_reg_239 <= select_ln616_1_fu_199_p3;
    end
end

always @ (*) begin
    if ((icmp_ln610_fu_155_p2 == 1'd1)) begin
        ap_condition_pp0_exit_iter0_state2 = 1'b1;
    end else begin
        ap_condition_pp0_exit_iter0_state2 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if ((icmp_ln610_reg_230 == 1'd0)) begin
        if ((select_ln616_1_reg_239 == 1'd0)) begin
            ap_phi_mux_tmp_V_phi_fu_146_p4 = fifo_C_drain_in_V_V_dout;
        end else if ((select_ln616_1_reg_239 == 1'd1)) begin
            ap_phi_mux_tmp_V_phi_fu_146_p4 = local_C_0_V_q0;
        end else begin
            ap_phi_mux_tmp_V_phi_fu_146_p4 = ap_phi_reg_pp0_iter1_tmp_V_reg_143;
        end
    end else begin
        ap_phi_mux_tmp_V_phi_fu_146_p4 = ap_phi_reg_pp0_iter1_tmp_V_reg_143;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((select_ln616_1_reg_239 == 1'd0) & (icmp_ln610_reg_230 == 1'd0) & (1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_C_drain_in_V_V_blk_n = fifo_C_drain_in_V_V_empty_n;
    end else begin
        fifo_C_drain_in_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_predicate_op31_read_state3 == 1'b1))) begin
        fifo_C_drain_in_V_V_read = 1'b1;
    end else begin
        fifo_C_drain_in_V_V_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln610_reg_230 == 1'd0) & (1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_C_drain_out_V_V_blk_n = fifo_C_drain_out_V_V_full_n;
    end else begin
        fifo_C_drain_out_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln610_reg_230 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_C_drain_out_V_V_write = 1'b1;
    end else begin
        fifo_C_drain_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        local_C_0_V_ce0 = 1'b1;
    end else begin
        local_C_0_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if (((ap_start == 1'b1) & (en_read_read_fu_78_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_pp0_stage0 : begin
            if (~((icmp_ln610_fu_155_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if (((icmp_ln610_fu_155_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln610_fu_161_p2 = (indvar_flatten_reg_110 + 3'd1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd2];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((fifo_C_drain_in_V_V_empty_n == 1'b0) & (ap_predicate_op31_read_state3 == 1'b1)) | ((icmp_ln610_reg_230 == 1'd0) & (fifo_C_drain_out_V_V_full_n == 1'b0))));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((fifo_C_drain_in_V_V_empty_n == 1'b0) & (ap_predicate_op31_read_state3 == 1'b1)) | ((icmp_ln610_reg_230 == 1'd0) & (fifo_C_drain_out_V_V_full_n == 1'b0))));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((fifo_C_drain_in_V_V_empty_n == 1'b0) & (ap_predicate_op31_read_state3 == 1'b1)) | ((icmp_ln610_reg_230 == 1'd0) & (fifo_C_drain_out_V_V_full_n == 1'b0))));
end

assign ap_block_state2_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage0_iter1 = (((fifo_C_drain_in_V_V_empty_n == 1'b0) & (ap_predicate_op31_read_state3 == 1'b1)) | ((icmp_ln610_reg_230 == 1'd0) & (fifo_C_drain_out_V_V_full_n == 1'b0)));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_phi_reg_pp0_iter1_tmp_V_reg_143 = 'bx;

always @ (*) begin
    ap_predicate_op31_read_state3 = ((select_ln616_1_reg_239 == 1'd0) & (icmp_ln610_reg_230 == 1'd0));
end

assign c4_fu_167_p2 = (c4_0_reg_121 + 2'd1);

assign c5_fu_220_p2 = (select_ln616_fu_179_p3 + 2'd1);

assign en_read_read_fu_78_p2 = en;

assign fifo_C_drain_out_V_V_din = ap_phi_mux_tmp_V_phi_fu_146_p4;

assign icmp_ln610_fu_155_p2 = ((indvar_flatten_reg_110 == 3'd4) ? 1'b1 : 1'b0);

assign icmp_ln612_fu_173_p2 = ((c5_0_reg_132 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln616_1_fu_193_p2 = ((c4_0_reg_121 == 2'd1) ? 1'b1 : 1'b0);

assign icmp_ln616_fu_187_p2 = ((c4_0_reg_121 == 2'd0) ? 1'b1 : 1'b0);

assign local_C_0_V_address0 = zext_ln617_fu_215_p1;

assign select_ln610_fu_207_p3 = ((icmp_ln612_fu_173_p2[0:0] === 1'b1) ? c4_fu_167_p2 : c4_0_reg_121);

assign select_ln616_1_fu_199_p3 = ((icmp_ln612_fu_173_p2[0:0] === 1'b1) ? icmp_ln616_fu_187_p2 : icmp_ln616_1_fu_193_p2);

assign select_ln616_fu_179_p3 = ((icmp_ln612_fu_173_p2[0:0] === 1'b1) ? 2'd0 : c5_0_reg_132);

assign zext_ln617_fu_215_p1 = select_ln616_fu_179_p3;

endmodule //C_drain_IO_L1_out_in_2
