// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module B_IO_L2_in_inter_tra_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        local_B_0_V_address0,
        local_B_0_V_ce0,
        local_B_0_V_we0,
        local_B_0_V_d0,
        fifo_B_in_V_V_dout,
        fifo_B_in_V_V_empty_n,
        fifo_B_in_V_V_read,
        fifo_B_out_V_V_din,
        fifo_B_out_V_V_full_n,
        fifo_B_out_V_V_write
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
output  [0:0] local_B_0_V_address0;
output   local_B_0_V_ce0;
output   local_B_0_V_we0;
output  [127:0] local_B_0_V_d0;
input  [127:0] fifo_B_in_V_V_dout;
input   fifo_B_in_V_V_empty_n;
output   fifo_B_in_V_V_read;
output  [127:0] fifo_B_out_V_V_din;
input   fifo_B_out_V_V_full_n;
output   fifo_B_out_V_V_write;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg local_B_0_V_ce0;
reg local_B_0_V_we0;
reg fifo_B_in_V_V_read;
reg fifo_B_out_V_V_write;

(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    fifo_B_in_V_V_blk_n;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
wire    ap_block_pp0_stage0;
reg   [0:0] icmp_ln288_reg_205;
reg    fifo_B_out_V_V_blk_n;
reg   [0:0] select_ln295_1_reg_219;
reg   [2:0] indvar_flatten_reg_102;
reg   [1:0] c3_0_reg_113;
reg   [1:0] c4_0_reg_124;
wire   [0:0] icmp_ln288_fu_135_p2;
wire    ap_block_state2_pp0_stage0_iter0;
reg    ap_block_state3_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_11001;
wire   [2:0] add_ln288_fu_141_p2;
reg    ap_enable_reg_pp0_iter0;
wire   [1:0] select_ln295_fu_159_p3;
reg   [1:0] select_ln295_reg_214;
wire   [0:0] select_ln295_1_fu_179_p3;
wire   [1:0] select_ln288_fu_187_p3;
wire   [1:0] c4_fu_195_p2;
reg    ap_block_pp0_stage0_subdone;
reg    ap_condition_pp0_exit_iter0_state2;
wire   [63:0] zext_ln296_fu_201_p1;
reg    ap_block_pp0_stage0_01001;
wire   [0:0] icmp_ln290_fu_153_p2;
wire   [1:0] c3_fu_147_p2;
wire   [0:0] icmp_ln295_fu_167_p2;
wire   [0:0] icmp_ln295_1_fu_173_p2;
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
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b1 == ap_condition_pp0_exit_iter0_state2))) begin
            ap_enable_reg_pp0_iter0 <= 1'b0;
        end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter0 <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_condition_pp0_exit_iter0_state2))) begin
            ap_enable_reg_pp0_iter1 <= (1'b1 ^ ap_condition_pp0_exit_iter0_state2);
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln288_fu_135_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        c3_0_reg_113 <= select_ln288_fu_187_p3;
    end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        c3_0_reg_113 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln288_fu_135_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        c4_0_reg_124 <= c4_fu_195_p2;
    end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        c4_0_reg_124 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln288_fu_135_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        indvar_flatten_reg_102 <= add_ln288_fu_141_p2;
    end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        indvar_flatten_reg_102 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        icmp_ln288_reg_205 <= icmp_ln288_fu_135_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln288_fu_135_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        select_ln295_1_reg_219 <= select_ln295_1_fu_179_p3;
        select_ln295_reg_214 <= select_ln295_fu_159_p3;
    end
end

always @ (*) begin
    if ((icmp_ln288_fu_135_p2 == 1'd1)) begin
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
    if ((1'b1 == ap_CS_fsm_state4)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln288_reg_205 == 1'd0) & (1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_B_in_V_V_blk_n = fifo_B_in_V_V_empty_n;
    end else begin
        fifo_B_in_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (icmp_ln288_reg_205 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_B_in_V_V_read = 1'b1;
    end else begin
        fifo_B_in_V_V_read = 1'b0;
    end
end

always @ (*) begin
    if (((select_ln295_1_reg_219 == 1'd0) & (1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_B_out_V_V_blk_n = fifo_B_out_V_V_full_n;
    end else begin
        fifo_B_out_V_V_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (select_ln295_1_reg_219 == 1'd0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        fifo_B_out_V_V_write = 1'b1;
    end else begin
        fifo_B_out_V_V_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        local_B_0_V_ce0 = 1'b1;
    end else begin
        local_B_0_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (select_ln295_1_reg_219 == 1'd1) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        local_B_0_V_we0 = 1'b1;
    end else begin
        local_B_0_V_we0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_pp0_stage0 : begin
            if (~((1'b0 == ap_block_pp0_stage0_subdone) & (icmp_ln288_fu_135_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (icmp_ln288_fu_135_p2 == 1'd1) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
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

assign add_ln288_fu_141_p2 = (indvar_flatten_reg_102 + 3'd1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd2];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((select_ln295_1_reg_219 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln288_reg_205 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0))));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((select_ln295_1_reg_219 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln288_reg_205 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0))));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((ap_enable_reg_pp0_iter1 == 1'b1) & (((select_ln295_1_reg_219 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln288_reg_205 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0))));
end

assign ap_block_state2_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage0_iter1 = (((select_ln295_1_reg_219 == 1'd0) & (fifo_B_out_V_V_full_n == 1'b0)) | ((icmp_ln288_reg_205 == 1'd0) & (fifo_B_in_V_V_empty_n == 1'b0)));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign c3_fu_147_p2 = (c3_0_reg_113 + 2'd1);

assign c4_fu_195_p2 = (select_ln295_fu_159_p3 + 2'd1);

assign fifo_B_out_V_V_din = fifo_B_in_V_V_dout;

assign icmp_ln288_fu_135_p2 = ((indvar_flatten_reg_102 == 3'd4) ? 1'b1 : 1'b0);

assign icmp_ln290_fu_153_p2 = ((c4_0_reg_124 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln295_1_fu_173_p2 = ((c3_0_reg_113 == 2'd0) ? 1'b1 : 1'b0);

assign icmp_ln295_fu_167_p2 = ((c3_fu_147_p2 == 2'd0) ? 1'b1 : 1'b0);

assign local_B_0_V_address0 = zext_ln296_fu_201_p1;

assign local_B_0_V_d0 = fifo_B_in_V_V_dout;

assign select_ln288_fu_187_p3 = ((icmp_ln290_fu_153_p2[0:0] === 1'b1) ? c3_fu_147_p2 : c3_0_reg_113);

assign select_ln295_1_fu_179_p3 = ((icmp_ln290_fu_153_p2[0:0] === 1'b1) ? icmp_ln295_fu_167_p2 : icmp_ln295_1_fu_173_p2);

assign select_ln295_fu_159_p3 = ((icmp_ln290_fu_153_p2[0:0] === 1'b1) ? 2'd0 : c4_0_reg_124);

assign zext_ln296_fu_201_p1 = select_ln295_reg_214;

endmodule //B_IO_L2_in_inter_tra_1
