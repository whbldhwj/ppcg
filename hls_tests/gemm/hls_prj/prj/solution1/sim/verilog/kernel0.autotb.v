// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
 `timescale 1ns/1ps


`define AUTOTB_DUT      kernel0
`define AUTOTB_DUT_INST AESL_inst_kernel0
`define AUTOTB_TOP      apatb_kernel0_top
`define AUTOTB_LAT_RESULT_FILE "kernel0.result.lat.rb"
`define AUTOTB_PER_RESULT_TRANS_FILE "kernel0.performance.result.transaction.xml"
`define AUTOTB_TOP_INST AESL_inst_apatb_kernel0_top
`define AUTOTB_MAX_ALLOW_LATENCY  15000000
`define AUTOTB_CLOCK_PERIOD_DIV2 2.50

`define AESL_DEPTH_gmem_A 1
`define AESL_DEPTH_gmem_B 1
`define AESL_DEPTH_gmem_C 1
`define AESL_DEPTH_A_V 1
`define AESL_DEPTH_B_V 1
`define AESL_DEPTH_C_V 1
`define AUTOTB_TVIN_gmem_A  "../tv/cdatafile/c.kernel0.autotvin_gmem_A.dat"
`define AUTOTB_TVIN_gmem_B  "../tv/cdatafile/c.kernel0.autotvin_gmem_B.dat"
`define AUTOTB_TVIN_A_V  "../tv/cdatafile/c.kernel0.autotvin_A_V.dat"
`define AUTOTB_TVIN_B_V  "../tv/cdatafile/c.kernel0.autotvin_B_V.dat"
`define AUTOTB_TVIN_C_V  "../tv/cdatafile/c.kernel0.autotvin_C_V.dat"
`define AUTOTB_TVIN_gmem_A_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvin_gmem_A.dat"
`define AUTOTB_TVIN_gmem_B_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvin_gmem_B.dat"
`define AUTOTB_TVIN_A_V_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvin_A_V.dat"
`define AUTOTB_TVIN_B_V_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvin_B_V.dat"
`define AUTOTB_TVIN_C_V_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvin_C_V.dat"
`define AUTOTB_TVOUT_gmem_C  "../tv/cdatafile/c.kernel0.autotvout_gmem_C.dat"
`define AUTOTB_TVOUT_gmem_C_out_wrapc  "../tv/rtldatafile/rtl.kernel0.autotvout_gmem_C.dat"
module `AUTOTB_TOP;

parameter AUTOTB_TRANSACTION_NUM = 1;
parameter PROGRESS_TIMEOUT = 10000000;
parameter LATENCY_ESTIMATION = 147;
parameter LENGTH_gmem_A = 16;
parameter LENGTH_gmem_B = 16;
parameter LENGTH_gmem_C = 32;
parameter LENGTH_A_V = 1;
parameter LENGTH_B_V = 1;
parameter LENGTH_C_V = 1;

task read_token;
    input integer fp;
    output reg [279 : 0] token;
    integer ret;
    begin
        token = "";
        ret = 0;
        ret = $fscanf(fp,"%s",token);
    end
endtask

reg AESL_clock;
reg rst;
reg start;
reg ce;
reg tb_continue;
wire AESL_start;
wire AESL_reset;
wire AESL_ce;
wire AESL_ready;
wire AESL_idle;
wire AESL_continue;
wire AESL_done;
reg AESL_done_delay = 0;
reg AESL_done_delay2 = 0;
reg AESL_ready_delay = 0;
wire ready;
wire ready_wire;
wire [5 : 0] control_AWADDR;
wire  control_AWVALID;
wire  control_AWREADY;
wire  control_WVALID;
wire  control_WREADY;
wire [31 : 0] control_WDATA;
wire [3 : 0] control_WSTRB;
wire [5 : 0] control_ARADDR;
wire  control_ARVALID;
wire  control_ARREADY;
wire  control_RVALID;
wire  control_RREADY;
wire [31 : 0] control_RDATA;
wire [1 : 0] control_RRESP;
wire  control_BVALID;
wire  control_BREADY;
wire [1 : 0] control_BRESP;
wire  control_INTERRUPT;
wire  gmem_A_AWVALID;
wire  gmem_A_AWREADY;
wire [31 : 0] gmem_A_AWADDR;
wire [0 : 0] gmem_A_AWID;
wire [7 : 0] gmem_A_AWLEN;
wire [2 : 0] gmem_A_AWSIZE;
wire [1 : 0] gmem_A_AWBURST;
wire [1 : 0] gmem_A_AWLOCK;
wire [3 : 0] gmem_A_AWCACHE;
wire [2 : 0] gmem_A_AWPROT;
wire [3 : 0] gmem_A_AWQOS;
wire [3 : 0] gmem_A_AWREGION;
wire [0 : 0] gmem_A_AWUSER;
wire  gmem_A_WVALID;
wire  gmem_A_WREADY;
wire [127 : 0] gmem_A_WDATA;
wire [15 : 0] gmem_A_WSTRB;
wire  gmem_A_WLAST;
wire [0 : 0] gmem_A_WID;
wire [0 : 0] gmem_A_WUSER;
wire  gmem_A_ARVALID;
wire  gmem_A_ARREADY;
wire [31 : 0] gmem_A_ARADDR;
wire [0 : 0] gmem_A_ARID;
wire [7 : 0] gmem_A_ARLEN;
wire [2 : 0] gmem_A_ARSIZE;
wire [1 : 0] gmem_A_ARBURST;
wire [1 : 0] gmem_A_ARLOCK;
wire [3 : 0] gmem_A_ARCACHE;
wire [2 : 0] gmem_A_ARPROT;
wire [3 : 0] gmem_A_ARQOS;
wire [3 : 0] gmem_A_ARREGION;
wire [0 : 0] gmem_A_ARUSER;
wire  gmem_A_RVALID;
wire  gmem_A_RREADY;
wire [127 : 0] gmem_A_RDATA;
wire  gmem_A_RLAST;
wire [0 : 0] gmem_A_RID;
wire [0 : 0] gmem_A_RUSER;
wire [1 : 0] gmem_A_RRESP;
wire  gmem_A_BVALID;
wire  gmem_A_BREADY;
wire [1 : 0] gmem_A_BRESP;
wire [0 : 0] gmem_A_BID;
wire [0 : 0] gmem_A_BUSER;
wire  gmem_B_AWVALID;
wire  gmem_B_AWREADY;
wire [31 : 0] gmem_B_AWADDR;
wire [0 : 0] gmem_B_AWID;
wire [7 : 0] gmem_B_AWLEN;
wire [2 : 0] gmem_B_AWSIZE;
wire [1 : 0] gmem_B_AWBURST;
wire [1 : 0] gmem_B_AWLOCK;
wire [3 : 0] gmem_B_AWCACHE;
wire [2 : 0] gmem_B_AWPROT;
wire [3 : 0] gmem_B_AWQOS;
wire [3 : 0] gmem_B_AWREGION;
wire [0 : 0] gmem_B_AWUSER;
wire  gmem_B_WVALID;
wire  gmem_B_WREADY;
wire [127 : 0] gmem_B_WDATA;
wire [15 : 0] gmem_B_WSTRB;
wire  gmem_B_WLAST;
wire [0 : 0] gmem_B_WID;
wire [0 : 0] gmem_B_WUSER;
wire  gmem_B_ARVALID;
wire  gmem_B_ARREADY;
wire [31 : 0] gmem_B_ARADDR;
wire [0 : 0] gmem_B_ARID;
wire [7 : 0] gmem_B_ARLEN;
wire [2 : 0] gmem_B_ARSIZE;
wire [1 : 0] gmem_B_ARBURST;
wire [1 : 0] gmem_B_ARLOCK;
wire [3 : 0] gmem_B_ARCACHE;
wire [2 : 0] gmem_B_ARPROT;
wire [3 : 0] gmem_B_ARQOS;
wire [3 : 0] gmem_B_ARREGION;
wire [0 : 0] gmem_B_ARUSER;
wire  gmem_B_RVALID;
wire  gmem_B_RREADY;
wire [127 : 0] gmem_B_RDATA;
wire  gmem_B_RLAST;
wire [0 : 0] gmem_B_RID;
wire [0 : 0] gmem_B_RUSER;
wire [1 : 0] gmem_B_RRESP;
wire  gmem_B_BVALID;
wire  gmem_B_BREADY;
wire [1 : 0] gmem_B_BRESP;
wire [0 : 0] gmem_B_BID;
wire [0 : 0] gmem_B_BUSER;
wire  gmem_C_AWVALID;
wire  gmem_C_AWREADY;
wire [31 : 0] gmem_C_AWADDR;
wire [0 : 0] gmem_C_AWID;
wire [7 : 0] gmem_C_AWLEN;
wire [2 : 0] gmem_C_AWSIZE;
wire [1 : 0] gmem_C_AWBURST;
wire [1 : 0] gmem_C_AWLOCK;
wire [3 : 0] gmem_C_AWCACHE;
wire [2 : 0] gmem_C_AWPROT;
wire [3 : 0] gmem_C_AWQOS;
wire [3 : 0] gmem_C_AWREGION;
wire [0 : 0] gmem_C_AWUSER;
wire  gmem_C_WVALID;
wire  gmem_C_WREADY;
wire [63 : 0] gmem_C_WDATA;
wire [7 : 0] gmem_C_WSTRB;
wire  gmem_C_WLAST;
wire [0 : 0] gmem_C_WID;
wire [0 : 0] gmem_C_WUSER;
wire  gmem_C_ARVALID;
wire  gmem_C_ARREADY;
wire [31 : 0] gmem_C_ARADDR;
wire [0 : 0] gmem_C_ARID;
wire [7 : 0] gmem_C_ARLEN;
wire [2 : 0] gmem_C_ARSIZE;
wire [1 : 0] gmem_C_ARBURST;
wire [1 : 0] gmem_C_ARLOCK;
wire [3 : 0] gmem_C_ARCACHE;
wire [2 : 0] gmem_C_ARPROT;
wire [3 : 0] gmem_C_ARQOS;
wire [3 : 0] gmem_C_ARREGION;
wire [0 : 0] gmem_C_ARUSER;
wire  gmem_C_RVALID;
wire  gmem_C_RREADY;
wire [63 : 0] gmem_C_RDATA;
wire  gmem_C_RLAST;
wire [0 : 0] gmem_C_RID;
wire [0 : 0] gmem_C_RUSER;
wire [1 : 0] gmem_C_RRESP;
wire  gmem_C_BVALID;
wire  gmem_C_BREADY;
wire [1 : 0] gmem_C_BRESP;
wire [0 : 0] gmem_C_BID;
wire [0 : 0] gmem_C_BUSER;
integer done_cnt = 0;
integer AESL_ready_cnt = 0;
integer ready_cnt = 0;
reg ready_initial;
reg ready_initial_n;
reg ready_last_n;
reg ready_delay_last_n;
reg done_delay_last_n;
reg interface_done = 0;
wire control_write_data_finish;
wire AESL_slave_start;
reg AESL_slave_start_lock = 0;
wire AESL_slave_write_start_in;
wire AESL_slave_write_start_finish;
reg AESL_slave_ready;
wire AESL_slave_output_done;
wire AESL_slave_done;
reg ready_rise = 0;
reg start_rise = 0;
reg slave_start_status = 0;
reg slave_done_status = 0;
reg ap_done_lock = 0;

wire ap_clk;
wire ap_rst_n;
wire ap_rst_n_n;

`AUTOTB_DUT `AUTOTB_DUT_INST(
    .s_axi_control_AWADDR(control_AWADDR),
    .s_axi_control_AWVALID(control_AWVALID),
    .s_axi_control_AWREADY(control_AWREADY),
    .s_axi_control_WVALID(control_WVALID),
    .s_axi_control_WREADY(control_WREADY),
    .s_axi_control_WDATA(control_WDATA),
    .s_axi_control_WSTRB(control_WSTRB),
    .s_axi_control_ARADDR(control_ARADDR),
    .s_axi_control_ARVALID(control_ARVALID),
    .s_axi_control_ARREADY(control_ARREADY),
    .s_axi_control_RVALID(control_RVALID),
    .s_axi_control_RREADY(control_RREADY),
    .s_axi_control_RDATA(control_RDATA),
    .s_axi_control_RRESP(control_RRESP),
    .s_axi_control_BVALID(control_BVALID),
    .s_axi_control_BREADY(control_BREADY),
    .s_axi_control_BRESP(control_BRESP),
    .interrupt(control_INTERRUPT),
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .m_axi_gmem_A_AWVALID(gmem_A_AWVALID),
    .m_axi_gmem_A_AWREADY(gmem_A_AWREADY),
    .m_axi_gmem_A_AWADDR(gmem_A_AWADDR),
    .m_axi_gmem_A_AWID(gmem_A_AWID),
    .m_axi_gmem_A_AWLEN(gmem_A_AWLEN),
    .m_axi_gmem_A_AWSIZE(gmem_A_AWSIZE),
    .m_axi_gmem_A_AWBURST(gmem_A_AWBURST),
    .m_axi_gmem_A_AWLOCK(gmem_A_AWLOCK),
    .m_axi_gmem_A_AWCACHE(gmem_A_AWCACHE),
    .m_axi_gmem_A_AWPROT(gmem_A_AWPROT),
    .m_axi_gmem_A_AWQOS(gmem_A_AWQOS),
    .m_axi_gmem_A_AWREGION(gmem_A_AWREGION),
    .m_axi_gmem_A_AWUSER(gmem_A_AWUSER),
    .m_axi_gmem_A_WVALID(gmem_A_WVALID),
    .m_axi_gmem_A_WREADY(gmem_A_WREADY),
    .m_axi_gmem_A_WDATA(gmem_A_WDATA),
    .m_axi_gmem_A_WSTRB(gmem_A_WSTRB),
    .m_axi_gmem_A_WLAST(gmem_A_WLAST),
    .m_axi_gmem_A_WID(gmem_A_WID),
    .m_axi_gmem_A_WUSER(gmem_A_WUSER),
    .m_axi_gmem_A_ARVALID(gmem_A_ARVALID),
    .m_axi_gmem_A_ARREADY(gmem_A_ARREADY),
    .m_axi_gmem_A_ARADDR(gmem_A_ARADDR),
    .m_axi_gmem_A_ARID(gmem_A_ARID),
    .m_axi_gmem_A_ARLEN(gmem_A_ARLEN),
    .m_axi_gmem_A_ARSIZE(gmem_A_ARSIZE),
    .m_axi_gmem_A_ARBURST(gmem_A_ARBURST),
    .m_axi_gmem_A_ARLOCK(gmem_A_ARLOCK),
    .m_axi_gmem_A_ARCACHE(gmem_A_ARCACHE),
    .m_axi_gmem_A_ARPROT(gmem_A_ARPROT),
    .m_axi_gmem_A_ARQOS(gmem_A_ARQOS),
    .m_axi_gmem_A_ARREGION(gmem_A_ARREGION),
    .m_axi_gmem_A_ARUSER(gmem_A_ARUSER),
    .m_axi_gmem_A_RVALID(gmem_A_RVALID),
    .m_axi_gmem_A_RREADY(gmem_A_RREADY),
    .m_axi_gmem_A_RDATA(gmem_A_RDATA),
    .m_axi_gmem_A_RLAST(gmem_A_RLAST),
    .m_axi_gmem_A_RID(gmem_A_RID),
    .m_axi_gmem_A_RUSER(gmem_A_RUSER),
    .m_axi_gmem_A_RRESP(gmem_A_RRESP),
    .m_axi_gmem_A_BVALID(gmem_A_BVALID),
    .m_axi_gmem_A_BREADY(gmem_A_BREADY),
    .m_axi_gmem_A_BRESP(gmem_A_BRESP),
    .m_axi_gmem_A_BID(gmem_A_BID),
    .m_axi_gmem_A_BUSER(gmem_A_BUSER),
    .m_axi_gmem_B_AWVALID(gmem_B_AWVALID),
    .m_axi_gmem_B_AWREADY(gmem_B_AWREADY),
    .m_axi_gmem_B_AWADDR(gmem_B_AWADDR),
    .m_axi_gmem_B_AWID(gmem_B_AWID),
    .m_axi_gmem_B_AWLEN(gmem_B_AWLEN),
    .m_axi_gmem_B_AWSIZE(gmem_B_AWSIZE),
    .m_axi_gmem_B_AWBURST(gmem_B_AWBURST),
    .m_axi_gmem_B_AWLOCK(gmem_B_AWLOCK),
    .m_axi_gmem_B_AWCACHE(gmem_B_AWCACHE),
    .m_axi_gmem_B_AWPROT(gmem_B_AWPROT),
    .m_axi_gmem_B_AWQOS(gmem_B_AWQOS),
    .m_axi_gmem_B_AWREGION(gmem_B_AWREGION),
    .m_axi_gmem_B_AWUSER(gmem_B_AWUSER),
    .m_axi_gmem_B_WVALID(gmem_B_WVALID),
    .m_axi_gmem_B_WREADY(gmem_B_WREADY),
    .m_axi_gmem_B_WDATA(gmem_B_WDATA),
    .m_axi_gmem_B_WSTRB(gmem_B_WSTRB),
    .m_axi_gmem_B_WLAST(gmem_B_WLAST),
    .m_axi_gmem_B_WID(gmem_B_WID),
    .m_axi_gmem_B_WUSER(gmem_B_WUSER),
    .m_axi_gmem_B_ARVALID(gmem_B_ARVALID),
    .m_axi_gmem_B_ARREADY(gmem_B_ARREADY),
    .m_axi_gmem_B_ARADDR(gmem_B_ARADDR),
    .m_axi_gmem_B_ARID(gmem_B_ARID),
    .m_axi_gmem_B_ARLEN(gmem_B_ARLEN),
    .m_axi_gmem_B_ARSIZE(gmem_B_ARSIZE),
    .m_axi_gmem_B_ARBURST(gmem_B_ARBURST),
    .m_axi_gmem_B_ARLOCK(gmem_B_ARLOCK),
    .m_axi_gmem_B_ARCACHE(gmem_B_ARCACHE),
    .m_axi_gmem_B_ARPROT(gmem_B_ARPROT),
    .m_axi_gmem_B_ARQOS(gmem_B_ARQOS),
    .m_axi_gmem_B_ARREGION(gmem_B_ARREGION),
    .m_axi_gmem_B_ARUSER(gmem_B_ARUSER),
    .m_axi_gmem_B_RVALID(gmem_B_RVALID),
    .m_axi_gmem_B_RREADY(gmem_B_RREADY),
    .m_axi_gmem_B_RDATA(gmem_B_RDATA),
    .m_axi_gmem_B_RLAST(gmem_B_RLAST),
    .m_axi_gmem_B_RID(gmem_B_RID),
    .m_axi_gmem_B_RUSER(gmem_B_RUSER),
    .m_axi_gmem_B_RRESP(gmem_B_RRESP),
    .m_axi_gmem_B_BVALID(gmem_B_BVALID),
    .m_axi_gmem_B_BREADY(gmem_B_BREADY),
    .m_axi_gmem_B_BRESP(gmem_B_BRESP),
    .m_axi_gmem_B_BID(gmem_B_BID),
    .m_axi_gmem_B_BUSER(gmem_B_BUSER),
    .m_axi_gmem_C_AWVALID(gmem_C_AWVALID),
    .m_axi_gmem_C_AWREADY(gmem_C_AWREADY),
    .m_axi_gmem_C_AWADDR(gmem_C_AWADDR),
    .m_axi_gmem_C_AWID(gmem_C_AWID),
    .m_axi_gmem_C_AWLEN(gmem_C_AWLEN),
    .m_axi_gmem_C_AWSIZE(gmem_C_AWSIZE),
    .m_axi_gmem_C_AWBURST(gmem_C_AWBURST),
    .m_axi_gmem_C_AWLOCK(gmem_C_AWLOCK),
    .m_axi_gmem_C_AWCACHE(gmem_C_AWCACHE),
    .m_axi_gmem_C_AWPROT(gmem_C_AWPROT),
    .m_axi_gmem_C_AWQOS(gmem_C_AWQOS),
    .m_axi_gmem_C_AWREGION(gmem_C_AWREGION),
    .m_axi_gmem_C_AWUSER(gmem_C_AWUSER),
    .m_axi_gmem_C_WVALID(gmem_C_WVALID),
    .m_axi_gmem_C_WREADY(gmem_C_WREADY),
    .m_axi_gmem_C_WDATA(gmem_C_WDATA),
    .m_axi_gmem_C_WSTRB(gmem_C_WSTRB),
    .m_axi_gmem_C_WLAST(gmem_C_WLAST),
    .m_axi_gmem_C_WID(gmem_C_WID),
    .m_axi_gmem_C_WUSER(gmem_C_WUSER),
    .m_axi_gmem_C_ARVALID(gmem_C_ARVALID),
    .m_axi_gmem_C_ARREADY(gmem_C_ARREADY),
    .m_axi_gmem_C_ARADDR(gmem_C_ARADDR),
    .m_axi_gmem_C_ARID(gmem_C_ARID),
    .m_axi_gmem_C_ARLEN(gmem_C_ARLEN),
    .m_axi_gmem_C_ARSIZE(gmem_C_ARSIZE),
    .m_axi_gmem_C_ARBURST(gmem_C_ARBURST),
    .m_axi_gmem_C_ARLOCK(gmem_C_ARLOCK),
    .m_axi_gmem_C_ARCACHE(gmem_C_ARCACHE),
    .m_axi_gmem_C_ARPROT(gmem_C_ARPROT),
    .m_axi_gmem_C_ARQOS(gmem_C_ARQOS),
    .m_axi_gmem_C_ARREGION(gmem_C_ARREGION),
    .m_axi_gmem_C_ARUSER(gmem_C_ARUSER),
    .m_axi_gmem_C_RVALID(gmem_C_RVALID),
    .m_axi_gmem_C_RREADY(gmem_C_RREADY),
    .m_axi_gmem_C_RDATA(gmem_C_RDATA),
    .m_axi_gmem_C_RLAST(gmem_C_RLAST),
    .m_axi_gmem_C_RID(gmem_C_RID),
    .m_axi_gmem_C_RUSER(gmem_C_RUSER),
    .m_axi_gmem_C_RRESP(gmem_C_RRESP),
    .m_axi_gmem_C_BVALID(gmem_C_BVALID),
    .m_axi_gmem_C_BREADY(gmem_C_BREADY),
    .m_axi_gmem_C_BRESP(gmem_C_BRESP),
    .m_axi_gmem_C_BID(gmem_C_BID),
    .m_axi_gmem_C_BUSER(gmem_C_BUSER));

// Assignment for control signal
assign ap_clk = AESL_clock;
assign ap_rst_n = AESL_reset;
assign ap_rst_n_n = ~AESL_reset;
assign AESL_reset = rst;
assign AESL_start = start;
assign AESL_ce = ce;
assign AESL_continue = tb_continue;
  assign AESL_slave_write_start_in = slave_start_status  & control_write_data_finish;
  assign AESL_slave_start = AESL_slave_write_start_finish;
  assign AESL_done = slave_done_status ;

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
    begin
        slave_start_status <= 1;
    end
    else begin
        if (AESL_start == 1 ) begin
            start_rise = 1;
        end
        if (start_rise == 1 && AESL_done == 1 ) begin
            slave_start_status <= 1;
        end
        if (AESL_slave_write_start_in == 1 && AESL_done == 0) begin 
            slave_start_status <= 0;
            start_rise = 0;
        end
    end
end

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
    begin
        AESL_slave_ready <= 0;
        ready_rise = 0;
    end
    else begin
        if (AESL_ready == 1 ) begin
            ready_rise = 1;
        end
        if (ready_rise == 1 && AESL_done_delay == 1 ) begin
            AESL_slave_ready <= 1;
        end
        if (AESL_slave_ready == 1) begin 
            AESL_slave_ready <= 0;
            ready_rise = 0;
        end
    end
end

always @ (posedge AESL_clock)
begin
    if (AESL_done == 1) begin
        slave_done_status <= 0;
    end
    else if (AESL_slave_output_done == 1 ) begin
        slave_done_status <= 1;
    end
end






wire    AESL_axi_master_gmem_A_ready;
wire    AESL_axi_master_gmem_A_done;
wire [32 - 1:0] A_V;
AESL_axi_master_gmem_A AESL_AXI_MASTER_gmem_A(
    .clk   (AESL_clock),
    .reset (AESL_reset),
    .TRAN_gmem_A_AWVALID (gmem_A_AWVALID),
    .TRAN_gmem_A_AWREADY (gmem_A_AWREADY),
    .TRAN_gmem_A_AWADDR (gmem_A_AWADDR),
    .TRAN_gmem_A_AWID (gmem_A_AWID),
    .TRAN_gmem_A_AWLEN (gmem_A_AWLEN),
    .TRAN_gmem_A_AWSIZE (gmem_A_AWSIZE),
    .TRAN_gmem_A_AWBURST (gmem_A_AWBURST),
    .TRAN_gmem_A_AWLOCK (gmem_A_AWLOCK),
    .TRAN_gmem_A_AWCACHE (gmem_A_AWCACHE),
    .TRAN_gmem_A_AWPROT (gmem_A_AWPROT),
    .TRAN_gmem_A_AWQOS (gmem_A_AWQOS),
    .TRAN_gmem_A_AWREGION (gmem_A_AWREGION),
    .TRAN_gmem_A_AWUSER (gmem_A_AWUSER),
    .TRAN_gmem_A_WVALID (gmem_A_WVALID),
    .TRAN_gmem_A_WREADY (gmem_A_WREADY),
    .TRAN_gmem_A_WDATA (gmem_A_WDATA),
    .TRAN_gmem_A_WSTRB (gmem_A_WSTRB),
    .TRAN_gmem_A_WLAST (gmem_A_WLAST),
    .TRAN_gmem_A_WID (gmem_A_WID),
    .TRAN_gmem_A_WUSER (gmem_A_WUSER),
    .TRAN_gmem_A_ARVALID (gmem_A_ARVALID),
    .TRAN_gmem_A_ARREADY (gmem_A_ARREADY),
    .TRAN_gmem_A_ARADDR (gmem_A_ARADDR),
    .TRAN_gmem_A_ARID (gmem_A_ARID),
    .TRAN_gmem_A_ARLEN (gmem_A_ARLEN),
    .TRAN_gmem_A_ARSIZE (gmem_A_ARSIZE),
    .TRAN_gmem_A_ARBURST (gmem_A_ARBURST),
    .TRAN_gmem_A_ARLOCK (gmem_A_ARLOCK),
    .TRAN_gmem_A_ARCACHE (gmem_A_ARCACHE),
    .TRAN_gmem_A_ARPROT (gmem_A_ARPROT),
    .TRAN_gmem_A_ARQOS (gmem_A_ARQOS),
    .TRAN_gmem_A_ARREGION (gmem_A_ARREGION),
    .TRAN_gmem_A_ARUSER (gmem_A_ARUSER),
    .TRAN_gmem_A_RVALID (gmem_A_RVALID),
    .TRAN_gmem_A_RREADY (gmem_A_RREADY),
    .TRAN_gmem_A_RDATA (gmem_A_RDATA),
    .TRAN_gmem_A_RLAST (gmem_A_RLAST),
    .TRAN_gmem_A_RID (gmem_A_RID),
    .TRAN_gmem_A_RUSER (gmem_A_RUSER),
    .TRAN_gmem_A_RRESP (gmem_A_RRESP),
    .TRAN_gmem_A_BVALID (gmem_A_BVALID),
    .TRAN_gmem_A_BREADY (gmem_A_BREADY),
    .TRAN_gmem_A_BRESP (gmem_A_BRESP),
    .TRAN_gmem_A_BID (gmem_A_BID),
    .TRAN_gmem_A_BUSER (gmem_A_BUSER),
    .TRAN_gmem_A_A_V (A_V),
    .ready (AESL_axi_master_gmem_A_ready),
    .done  (AESL_axi_master_gmem_A_done)
);
assign    AESL_axi_master_gmem_A_ready    =   ready;
assign    AESL_axi_master_gmem_A_done    =   AESL_done_delay;
wire    AESL_axi_master_gmem_B_ready;
wire    AESL_axi_master_gmem_B_done;
wire [32 - 1:0] B_V;
AESL_axi_master_gmem_B AESL_AXI_MASTER_gmem_B(
    .clk   (AESL_clock),
    .reset (AESL_reset),
    .TRAN_gmem_B_AWVALID (gmem_B_AWVALID),
    .TRAN_gmem_B_AWREADY (gmem_B_AWREADY),
    .TRAN_gmem_B_AWADDR (gmem_B_AWADDR),
    .TRAN_gmem_B_AWID (gmem_B_AWID),
    .TRAN_gmem_B_AWLEN (gmem_B_AWLEN),
    .TRAN_gmem_B_AWSIZE (gmem_B_AWSIZE),
    .TRAN_gmem_B_AWBURST (gmem_B_AWBURST),
    .TRAN_gmem_B_AWLOCK (gmem_B_AWLOCK),
    .TRAN_gmem_B_AWCACHE (gmem_B_AWCACHE),
    .TRAN_gmem_B_AWPROT (gmem_B_AWPROT),
    .TRAN_gmem_B_AWQOS (gmem_B_AWQOS),
    .TRAN_gmem_B_AWREGION (gmem_B_AWREGION),
    .TRAN_gmem_B_AWUSER (gmem_B_AWUSER),
    .TRAN_gmem_B_WVALID (gmem_B_WVALID),
    .TRAN_gmem_B_WREADY (gmem_B_WREADY),
    .TRAN_gmem_B_WDATA (gmem_B_WDATA),
    .TRAN_gmem_B_WSTRB (gmem_B_WSTRB),
    .TRAN_gmem_B_WLAST (gmem_B_WLAST),
    .TRAN_gmem_B_WID (gmem_B_WID),
    .TRAN_gmem_B_WUSER (gmem_B_WUSER),
    .TRAN_gmem_B_ARVALID (gmem_B_ARVALID),
    .TRAN_gmem_B_ARREADY (gmem_B_ARREADY),
    .TRAN_gmem_B_ARADDR (gmem_B_ARADDR),
    .TRAN_gmem_B_ARID (gmem_B_ARID),
    .TRAN_gmem_B_ARLEN (gmem_B_ARLEN),
    .TRAN_gmem_B_ARSIZE (gmem_B_ARSIZE),
    .TRAN_gmem_B_ARBURST (gmem_B_ARBURST),
    .TRAN_gmem_B_ARLOCK (gmem_B_ARLOCK),
    .TRAN_gmem_B_ARCACHE (gmem_B_ARCACHE),
    .TRAN_gmem_B_ARPROT (gmem_B_ARPROT),
    .TRAN_gmem_B_ARQOS (gmem_B_ARQOS),
    .TRAN_gmem_B_ARREGION (gmem_B_ARREGION),
    .TRAN_gmem_B_ARUSER (gmem_B_ARUSER),
    .TRAN_gmem_B_RVALID (gmem_B_RVALID),
    .TRAN_gmem_B_RREADY (gmem_B_RREADY),
    .TRAN_gmem_B_RDATA (gmem_B_RDATA),
    .TRAN_gmem_B_RLAST (gmem_B_RLAST),
    .TRAN_gmem_B_RID (gmem_B_RID),
    .TRAN_gmem_B_RUSER (gmem_B_RUSER),
    .TRAN_gmem_B_RRESP (gmem_B_RRESP),
    .TRAN_gmem_B_BVALID (gmem_B_BVALID),
    .TRAN_gmem_B_BREADY (gmem_B_BREADY),
    .TRAN_gmem_B_BRESP (gmem_B_BRESP),
    .TRAN_gmem_B_BID (gmem_B_BID),
    .TRAN_gmem_B_BUSER (gmem_B_BUSER),
    .TRAN_gmem_B_B_V (B_V),
    .ready (AESL_axi_master_gmem_B_ready),
    .done  (AESL_axi_master_gmem_B_done)
);
assign    AESL_axi_master_gmem_B_ready    =   ready;
assign    AESL_axi_master_gmem_B_done    =   AESL_done_delay;
wire    AESL_axi_master_gmem_C_ready;
wire    AESL_axi_master_gmem_C_done;
wire [32 - 1:0] C_V;
AESL_axi_master_gmem_C AESL_AXI_MASTER_gmem_C(
    .clk   (AESL_clock),
    .reset (AESL_reset),
    .TRAN_gmem_C_AWVALID (gmem_C_AWVALID),
    .TRAN_gmem_C_AWREADY (gmem_C_AWREADY),
    .TRAN_gmem_C_AWADDR (gmem_C_AWADDR),
    .TRAN_gmem_C_AWID (gmem_C_AWID),
    .TRAN_gmem_C_AWLEN (gmem_C_AWLEN),
    .TRAN_gmem_C_AWSIZE (gmem_C_AWSIZE),
    .TRAN_gmem_C_AWBURST (gmem_C_AWBURST),
    .TRAN_gmem_C_AWLOCK (gmem_C_AWLOCK),
    .TRAN_gmem_C_AWCACHE (gmem_C_AWCACHE),
    .TRAN_gmem_C_AWPROT (gmem_C_AWPROT),
    .TRAN_gmem_C_AWQOS (gmem_C_AWQOS),
    .TRAN_gmem_C_AWREGION (gmem_C_AWREGION),
    .TRAN_gmem_C_AWUSER (gmem_C_AWUSER),
    .TRAN_gmem_C_WVALID (gmem_C_WVALID),
    .TRAN_gmem_C_WREADY (gmem_C_WREADY),
    .TRAN_gmem_C_WDATA (gmem_C_WDATA),
    .TRAN_gmem_C_WSTRB (gmem_C_WSTRB),
    .TRAN_gmem_C_WLAST (gmem_C_WLAST),
    .TRAN_gmem_C_WID (gmem_C_WID),
    .TRAN_gmem_C_WUSER (gmem_C_WUSER),
    .TRAN_gmem_C_ARVALID (gmem_C_ARVALID),
    .TRAN_gmem_C_ARREADY (gmem_C_ARREADY),
    .TRAN_gmem_C_ARADDR (gmem_C_ARADDR),
    .TRAN_gmem_C_ARID (gmem_C_ARID),
    .TRAN_gmem_C_ARLEN (gmem_C_ARLEN),
    .TRAN_gmem_C_ARSIZE (gmem_C_ARSIZE),
    .TRAN_gmem_C_ARBURST (gmem_C_ARBURST),
    .TRAN_gmem_C_ARLOCK (gmem_C_ARLOCK),
    .TRAN_gmem_C_ARCACHE (gmem_C_ARCACHE),
    .TRAN_gmem_C_ARPROT (gmem_C_ARPROT),
    .TRAN_gmem_C_ARQOS (gmem_C_ARQOS),
    .TRAN_gmem_C_ARREGION (gmem_C_ARREGION),
    .TRAN_gmem_C_ARUSER (gmem_C_ARUSER),
    .TRAN_gmem_C_RVALID (gmem_C_RVALID),
    .TRAN_gmem_C_RREADY (gmem_C_RREADY),
    .TRAN_gmem_C_RDATA (gmem_C_RDATA),
    .TRAN_gmem_C_RLAST (gmem_C_RLAST),
    .TRAN_gmem_C_RID (gmem_C_RID),
    .TRAN_gmem_C_RUSER (gmem_C_RUSER),
    .TRAN_gmem_C_RRESP (gmem_C_RRESP),
    .TRAN_gmem_C_BVALID (gmem_C_BVALID),
    .TRAN_gmem_C_BREADY (gmem_C_BREADY),
    .TRAN_gmem_C_BRESP (gmem_C_BRESP),
    .TRAN_gmem_C_BID (gmem_C_BID),
    .TRAN_gmem_C_BUSER (gmem_C_BUSER),
    .TRAN_gmem_C_C_V (C_V),
    .ready (AESL_axi_master_gmem_C_ready),
    .done  (AESL_axi_master_gmem_C_done)
);
assign    AESL_axi_master_gmem_C_ready    =   ready;
assign    AESL_axi_master_gmem_C_done    =   AESL_done_delay;

AESL_axi_slave_control AESL_AXI_SLAVE_control(
    .clk   (AESL_clock),
    .reset (AESL_reset),
    .TRAN_s_axi_control_AWADDR (control_AWADDR),
    .TRAN_s_axi_control_AWVALID (control_AWVALID),
    .TRAN_s_axi_control_AWREADY (control_AWREADY),
    .TRAN_s_axi_control_WVALID (control_WVALID),
    .TRAN_s_axi_control_WREADY (control_WREADY),
    .TRAN_s_axi_control_WDATA (control_WDATA),
    .TRAN_s_axi_control_WSTRB (control_WSTRB),
    .TRAN_s_axi_control_ARADDR (control_ARADDR),
    .TRAN_s_axi_control_ARVALID (control_ARVALID),
    .TRAN_s_axi_control_ARREADY (control_ARREADY),
    .TRAN_s_axi_control_RVALID (control_RVALID),
    .TRAN_s_axi_control_RREADY (control_RREADY),
    .TRAN_s_axi_control_RDATA (control_RDATA),
    .TRAN_s_axi_control_RRESP (control_RRESP),
    .TRAN_s_axi_control_BVALID (control_BVALID),
    .TRAN_s_axi_control_BREADY (control_BREADY),
    .TRAN_s_axi_control_BRESP (control_BRESP),
    .TRAN_control_interrupt (control_INTERRUPT),
    .TRAN_A_V (A_V),
    .TRAN_B_V (B_V),
    .TRAN_C_V (C_V),
    .TRAN_control_write_data_finish(control_write_data_finish),
    .TRAN_control_ready_out (AESL_ready),
    .TRAN_control_ready_in (AESL_slave_ready),
    .TRAN_control_done_out (AESL_slave_output_done),
    .TRAN_control_idle_out (AESL_idle),
    .TRAN_control_write_start_in     (AESL_slave_write_start_in),
    .TRAN_control_write_start_finish (AESL_slave_write_start_finish),
    .TRAN_control_transaction_done_in (AESL_done_delay),
    .TRAN_control_start_in  (AESL_slave_start)
);

initial begin : generate_AESL_ready_cnt_proc
    AESL_ready_cnt = 0;
    wait(AESL_reset === 1);
    while(AESL_ready_cnt != AUTOTB_TRANSACTION_NUM) begin
        while(AESL_ready !== 1) begin
            @(posedge AESL_clock);
            # 0.4;
        end
        @(negedge AESL_clock);
        AESL_ready_cnt = AESL_ready_cnt + 1;
        @(posedge AESL_clock);
        # 0.4;
    end
end

    event next_trigger_ready_cnt;
    
    initial begin : gen_ready_cnt
        ready_cnt = 0;
        wait (AESL_reset === 1);
        forever begin
            @ (posedge AESL_clock);
            if (ready == 1) begin
                if (ready_cnt < AUTOTB_TRANSACTION_NUM) begin
                    ready_cnt = ready_cnt + 1;
                end
            end
            -> next_trigger_ready_cnt;
        end
    end
    
    wire all_finish = (done_cnt == AUTOTB_TRANSACTION_NUM);
    
    // done_cnt
    always @ (posedge AESL_clock) begin
        if (~AESL_reset) begin
            done_cnt <= 0;
        end else begin
            if (AESL_done == 1) begin
                if (done_cnt < AUTOTB_TRANSACTION_NUM) begin
                    done_cnt <= done_cnt + 1;
                end
            end
        end
    end
    
    initial begin : finish_simulation
        wait (all_finish == 1);
        // last transaction is saved at negedge right after last done
        @ (posedge AESL_clock);
        @ (posedge AESL_clock);
        @ (posedge AESL_clock);
        @ (posedge AESL_clock);
        $finish;
    end
    
initial begin
    AESL_clock = 0;
    forever #`AUTOTB_CLOCK_PERIOD_DIV2 AESL_clock = ~AESL_clock;
end


reg end_gmem_A;
reg [31:0] size_gmem_A;
reg [31:0] size_gmem_A_backup;
reg end_gmem_B;
reg [31:0] size_gmem_B;
reg [31:0] size_gmem_B_backup;
reg end_A_V;
reg [31:0] size_A_V;
reg [31:0] size_A_V_backup;
reg end_B_V;
reg [31:0] size_B_V;
reg [31:0] size_B_V_backup;
reg end_C_V;
reg [31:0] size_C_V;
reg [31:0] size_C_V_backup;
reg end_gmem_C;
reg [31:0] size_gmem_C;
reg [31:0] size_gmem_C_backup;

initial begin : initial_process
    integer proc_rand;
    rst = 0;
    # 100;
    repeat(3) @ (posedge AESL_clock);
    rst = 1;
end
initial begin : start_process
    integer proc_rand;
    reg [31:0] start_cnt;
    ce = 1;
    start = 0;
    start_cnt = 0;
    wait (AESL_reset === 1);
    @ (posedge AESL_clock);
    #0 start = 1;
    start_cnt = start_cnt + 1;
    forever begin
        @ (posedge AESL_clock);
        if (start_cnt >= AUTOTB_TRANSACTION_NUM) begin
            // keep pushing garbage in
            #0 start = 1;
        end
        if (AESL_ready) begin
            start_cnt = start_cnt + 1;
        end
    end
end

always @(AESL_done)
begin
    tb_continue = AESL_done;
end

initial begin : ready_initial_process
    ready_initial = 0;
    wait (AESL_start === 1);
    ready_initial = 1;
    @(posedge AESL_clock);
    ready_initial = 0;
end

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
      AESL_ready_delay = 0;
  else
      AESL_ready_delay = AESL_ready;
end
initial begin : ready_last_n_process
  ready_last_n = 1;
  wait(ready_cnt == AUTOTB_TRANSACTION_NUM)
  @(posedge AESL_clock);
  ready_last_n <= 0;
end

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
      ready_delay_last_n = 0;
  else
      ready_delay_last_n <= ready_last_n;
end
assign ready = (ready_initial | AESL_ready_delay);
assign ready_wire = ready_initial | AESL_ready_delay;
initial begin : done_delay_last_n_process
  done_delay_last_n = 1;
  while(done_cnt < AUTOTB_TRANSACTION_NUM)
      @(posedge AESL_clock);
  # 0.1;
  done_delay_last_n = 0;
end

always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
  begin
      AESL_done_delay <= 0;
      AESL_done_delay2 <= 0;
  end
  else begin
      AESL_done_delay <= AESL_done & done_delay_last_n;
      AESL_done_delay2 <= AESL_done_delay;
  end
end
always @(posedge AESL_clock)
begin
    if(AESL_reset === 0)
      interface_done = 0;
  else begin
      # 0.01;
      if(ready === 1 && ready_cnt > 0 && ready_cnt < AUTOTB_TRANSACTION_NUM)
          interface_done = 1;
      else if(AESL_done_delay === 1 && done_cnt == AUTOTB_TRANSACTION_NUM)
          interface_done = 1;
      else
          interface_done = 0;
  end
end

reg dump_tvout_finish_gmem_C;

initial begin : dump_tvout_runtime_sign_gmem_C
    integer fp;
    dump_tvout_finish_gmem_C = 0;
    fp = $fopen(`AUTOTB_TVOUT_gmem_C_out_wrapc, "w");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_gmem_C_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[runtime]]]");
    $fclose(fp);
    wait (done_cnt == AUTOTB_TRANSACTION_NUM);
    // last transaction is saved at negedge right after last done
    @ (posedge AESL_clock);
    @ (posedge AESL_clock);
    @ (posedge AESL_clock);
    fp = $fopen(`AUTOTB_TVOUT_gmem_C_out_wrapc, "a");
    if (fp == 0) begin
        $display("Failed to open file \"%s\"!", `AUTOTB_TVOUT_gmem_C_out_wrapc);
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    $fdisplay(fp,"[[[/runtime]]]");
    $fclose(fp);
    dump_tvout_finish_gmem_C = 1;
end


////////////////////////////////////////////
// progress and performance
////////////////////////////////////////////

task wait_start();
    while (~AESL_start) begin
        @ (posedge AESL_clock);
    end
endtask

reg [31:0] clk_cnt = 0;
reg AESL_ready_p1;
reg AESL_start_p1;

always @ (posedge AESL_clock) begin
    clk_cnt <= clk_cnt + 1;
    AESL_ready_p1 <= AESL_ready;
    AESL_start_p1 <= AESL_start;
end

reg [31:0] start_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] start_cnt;
reg [31:0] ready_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] ap_ready_cnt;
reg [31:0] finish_timestamp [0:AUTOTB_TRANSACTION_NUM - 1];
reg [31:0] finish_cnt;
event report_progress;

initial begin
    start_cnt = 0;
    finish_cnt = 0;
    ap_ready_cnt = 0;
    wait (AESL_reset == 1);
    wait_start();
    start_timestamp[start_cnt] = clk_cnt;
    start_cnt = start_cnt + 1;
    if (AESL_done) begin
        finish_timestamp[finish_cnt] = clk_cnt;
        finish_cnt = finish_cnt + 1;
    end
    -> report_progress;
    forever begin
        @ (posedge AESL_clock);
        if (start_cnt < AUTOTB_TRANSACTION_NUM) begin
            if ((AESL_start && AESL_ready_p1)||(AESL_start && ~AESL_start_p1)) begin
                start_timestamp[start_cnt] = clk_cnt;
                start_cnt = start_cnt + 1;
            end
        end
        if (ap_ready_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_start_p1 && AESL_ready_p1) begin
                ready_timestamp[ap_ready_cnt] = clk_cnt;
                ap_ready_cnt = ap_ready_cnt + 1;
            end
        end
        if (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_done) begin
                finish_timestamp[finish_cnt] = clk_cnt;
                finish_cnt = finish_cnt + 1;
            end
        end
        -> report_progress;
    end
end

reg [31:0] progress_timeout;

initial begin : simulation_progress
    real intra_progress;
    wait (AESL_reset == 1);
    progress_timeout = PROGRESS_TIMEOUT;
    $display("////////////////////////////////////////////////////////////////////////////////////");
    $display("// Inter-Transaction Progress: Completed Transaction / Total Transaction");
    $display("// Intra-Transaction Progress: Measured Latency / Latency Estimation * 100%%");
    $display("//");
    $display("// RTL Simulation : \"Inter-Transaction Progress\" [\"Intra-Transaction Progress\"] @ \"Simulation Time\"");
    $display("////////////////////////////////////////////////////////////////////////////////////");
    print_progress();
    while (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
        @ (report_progress);
        if (finish_cnt < AUTOTB_TRANSACTION_NUM) begin
            if (AESL_done) begin
                print_progress();
                progress_timeout = PROGRESS_TIMEOUT;
            end else begin
                if (progress_timeout == 0) begin
                    print_progress();
                    progress_timeout = PROGRESS_TIMEOUT;
                end else begin
                    progress_timeout = progress_timeout - 1;
                end
            end
        end
    end
    print_progress();
    $display("////////////////////////////////////////////////////////////////////////////////////");
    calculate_performance();
end

task get_intra_progress(output real intra_progress);
    begin
        if (start_cnt > finish_cnt) begin
            intra_progress = clk_cnt - start_timestamp[finish_cnt];
        end else if(finish_cnt > 0) begin
            intra_progress = LATENCY_ESTIMATION;
        end else begin
            intra_progress = 0;
        end
        intra_progress = intra_progress / LATENCY_ESTIMATION;
    end
endtask

task print_progress();
    real intra_progress;
    begin
        if (LATENCY_ESTIMATION > 0) begin
            get_intra_progress(intra_progress);
            $display("// RTL Simulation : %0d / %0d [%2.2f%%] @ \"%0t\"", finish_cnt, AUTOTB_TRANSACTION_NUM, intra_progress * 100, $time);
        end else begin
            $display("// RTL Simulation : %0d / %0d [n/a] @ \"%0t\"", finish_cnt, AUTOTB_TRANSACTION_NUM, $time);
        end
    end
endtask

task calculate_performance();
    integer i;
    integer fp;
    reg [31:0] latency [0:AUTOTB_TRANSACTION_NUM - 1];
    reg [31:0] latency_min;
    reg [31:0] latency_max;
    reg [31:0] latency_total;
    reg [31:0] latency_average;
    reg [31:0] interval [0:AUTOTB_TRANSACTION_NUM - 2];
    reg [31:0] interval_min;
    reg [31:0] interval_max;
    reg [31:0] interval_total;
    reg [31:0] interval_average;
    begin
        latency_min = -1;
        latency_max = 0;
        latency_total = 0;
        interval_min = -1;
        interval_max = 0;
        interval_total = 0;

        for (i = 0; i < AUTOTB_TRANSACTION_NUM; i = i + 1) begin
            // calculate latency
            latency[i] = finish_timestamp[i] - start_timestamp[i];
            if (latency[i] > latency_max) latency_max = latency[i];
            if (latency[i] < latency_min) latency_min = latency[i];
            latency_total = latency_total + latency[i];
            // calculate interval
            if (AUTOTB_TRANSACTION_NUM == 1) begin
                interval[i] = 0;
                interval_max = 0;
                interval_min = 0;
                interval_total = 0;
            end else if (i < AUTOTB_TRANSACTION_NUM - 1) begin
                interval[i] = start_timestamp[i + 1] - start_timestamp[i];
                if (interval[i] > interval_max) interval_max = interval[i];
                if (interval[i] < interval_min) interval_min = interval[i];
                interval_total = interval_total + interval[i];
            end
        end

        latency_average = latency_total / AUTOTB_TRANSACTION_NUM;
        if (AUTOTB_TRANSACTION_NUM == 1) begin
            interval_average = 0;
        end else begin
            interval_average = interval_total / (AUTOTB_TRANSACTION_NUM - 1);
        end

        fp = $fopen(`AUTOTB_LAT_RESULT_FILE, "w");

        $fdisplay(fp, "$MAX_LATENCY = \"%0d\"", latency_max);
        $fdisplay(fp, "$MIN_LATENCY = \"%0d\"", latency_min);
        $fdisplay(fp, "$AVER_LATENCY = \"%0d\"", latency_average);
        $fdisplay(fp, "$MAX_THROUGHPUT = \"%0d\"", interval_max);
        $fdisplay(fp, "$MIN_THROUGHPUT = \"%0d\"", interval_min);
        $fdisplay(fp, "$AVER_THROUGHPUT = \"%0d\"", interval_average);

        $fclose(fp);

        fp = $fopen(`AUTOTB_PER_RESULT_TRANS_FILE, "w");

        $fdisplay(fp, "%20s%16s%16s", "", "latency", "interval");
        if (AUTOTB_TRANSACTION_NUM == 1) begin
            i = 0;
            $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
        end else begin
            for (i = 0; i < AUTOTB_TRANSACTION_NUM; i = i + 1) begin
                if (i < AUTOTB_TRANSACTION_NUM - 1) begin
                    $fdisplay(fp, "transaction%8d:%16d%16d", i, latency[i], interval[i]);
                end else begin
                    $fdisplay(fp, "transaction%8d:%16d               x", i, latency[i]);
                end
            end
        end

        $fclose(fp);
    end
endtask


////////////////////////////////////////////
// Dependence Check
////////////////////////////////////////////

`ifndef POST_SYN

`endif

AESL_deadlock_detector deadlock_detector(
    .reset(AESL_reset),
    .clock(AESL_clock));


endmodule
