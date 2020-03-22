set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName B_IO_L3_in
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {B_IO_L3_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ B_V int 128 regular {axi_master 0}  }
	{ B_V_offset int 32 regular {fifo 0}  }
	{ fifo_B_local_out_V_V int 128 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "B_V", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "B_V_offset", "interface" : "fifo", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_B_local_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 61
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ m_axi_B_V_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_B_V_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_B_V_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_B_V_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_B_V_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_B_V_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_B_V_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_B_V_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_B_V_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_B_V_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_B_V_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_B_V_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_B_V_BUSER sc_in sc_lv 1 signal 0 } 
	{ B_V_offset_dout sc_in sc_lv 32 signal 1 } 
	{ B_V_offset_empty_n sc_in sc_logic 1 signal 1 } 
	{ B_V_offset_read sc_out sc_logic 1 signal 1 } 
	{ fifo_B_local_out_V_V_din sc_out sc_lv 128 signal 2 } 
	{ fifo_B_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_B_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "m_axi_B_V_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWVALID" }} , 
 	{ "name": "m_axi_B_V_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWREADY" }} , 
 	{ "name": "m_axi_B_V_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWADDR" }} , 
 	{ "name": "m_axi_B_V_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWID" }} , 
 	{ "name": "m_axi_B_V_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "AWLEN" }} , 
 	{ "name": "m_axi_B_V_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_B_V_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWBURST" }} , 
 	{ "name": "m_axi_B_V_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_B_V_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_B_V_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "AWPROT" }} , 
 	{ "name": "m_axi_B_V_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWQOS" }} , 
 	{ "name": "m_axi_B_V_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "AWREGION" }} , 
 	{ "name": "m_axi_B_V_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "AWUSER" }} , 
 	{ "name": "m_axi_B_V_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WVALID" }} , 
 	{ "name": "m_axi_B_V_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WREADY" }} , 
 	{ "name": "m_axi_B_V_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "WDATA" }} , 
 	{ "name": "m_axi_B_V_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "B_V", "role": "WSTRB" }} , 
 	{ "name": "m_axi_B_V_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WLAST" }} , 
 	{ "name": "m_axi_B_V_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WID" }} , 
 	{ "name": "m_axi_B_V_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "WUSER" }} , 
 	{ "name": "m_axi_B_V_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARVALID" }} , 
 	{ "name": "m_axi_B_V_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARREADY" }} , 
 	{ "name": "m_axi_B_V_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARADDR" }} , 
 	{ "name": "m_axi_B_V_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARID" }} , 
 	{ "name": "m_axi_B_V_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V", "role": "ARLEN" }} , 
 	{ "name": "m_axi_B_V_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_B_V_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARBURST" }} , 
 	{ "name": "m_axi_B_V_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_B_V_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_B_V_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "B_V", "role": "ARPROT" }} , 
 	{ "name": "m_axi_B_V_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARQOS" }} , 
 	{ "name": "m_axi_B_V_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "B_V", "role": "ARREGION" }} , 
 	{ "name": "m_axi_B_V_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "ARUSER" }} , 
 	{ "name": "m_axi_B_V_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RVALID" }} , 
 	{ "name": "m_axi_B_V_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RREADY" }} , 
 	{ "name": "m_axi_B_V_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "B_V", "role": "RDATA" }} , 
 	{ "name": "m_axi_B_V_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RLAST" }} , 
 	{ "name": "m_axi_B_V_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RID" }} , 
 	{ "name": "m_axi_B_V_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "RUSER" }} , 
 	{ "name": "m_axi_B_V_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "RRESP" }} , 
 	{ "name": "m_axi_B_V_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BVALID" }} , 
 	{ "name": "m_axi_B_V_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BREADY" }} , 
 	{ "name": "m_axi_B_V_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "B_V", "role": "BRESP" }} , 
 	{ "name": "m_axi_B_V_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BID" }} , 
 	{ "name": "m_axi_B_V_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V", "role": "BUSER" }} , 
 	{ "name": "B_V_offset_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B_V_offset", "role": "dout" }} , 
 	{ "name": "B_V_offset_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "empty_n" }} , 
 	{ "name": "B_V_offset_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B_V_offset", "role": "read" }} , 
 	{ "name": "fifo_B_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_B_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_B_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_B_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "B_IO_L3_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "B_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "B_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "B_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	B_V { m_axi {  { m_axi_B_V_AWVALID VALID 1 1 }  { m_axi_B_V_AWREADY READY 0 1 }  { m_axi_B_V_AWADDR ADDR 1 32 }  { m_axi_B_V_AWID ID 1 1 }  { m_axi_B_V_AWLEN LEN 1 32 }  { m_axi_B_V_AWSIZE SIZE 1 3 }  { m_axi_B_V_AWBURST BURST 1 2 }  { m_axi_B_V_AWLOCK LOCK 1 2 }  { m_axi_B_V_AWCACHE CACHE 1 4 }  { m_axi_B_V_AWPROT PROT 1 3 }  { m_axi_B_V_AWQOS QOS 1 4 }  { m_axi_B_V_AWREGION REGION 1 4 }  { m_axi_B_V_AWUSER USER 1 1 }  { m_axi_B_V_WVALID VALID 1 1 }  { m_axi_B_V_WREADY READY 0 1 }  { m_axi_B_V_WDATA DATA 1 128 }  { m_axi_B_V_WSTRB STRB 1 16 }  { m_axi_B_V_WLAST LAST 1 1 }  { m_axi_B_V_WID ID 1 1 }  { m_axi_B_V_WUSER USER 1 1 }  { m_axi_B_V_ARVALID VALID 1 1 }  { m_axi_B_V_ARREADY READY 0 1 }  { m_axi_B_V_ARADDR ADDR 1 32 }  { m_axi_B_V_ARID ID 1 1 }  { m_axi_B_V_ARLEN LEN 1 32 }  { m_axi_B_V_ARSIZE SIZE 1 3 }  { m_axi_B_V_ARBURST BURST 1 2 }  { m_axi_B_V_ARLOCK LOCK 1 2 }  { m_axi_B_V_ARCACHE CACHE 1 4 }  { m_axi_B_V_ARPROT PROT 1 3 }  { m_axi_B_V_ARQOS QOS 1 4 }  { m_axi_B_V_ARREGION REGION 1 4 }  { m_axi_B_V_ARUSER USER 1 1 }  { m_axi_B_V_RVALID VALID 0 1 }  { m_axi_B_V_RREADY READY 1 1 }  { m_axi_B_V_RDATA DATA 0 128 }  { m_axi_B_V_RLAST LAST 0 1 }  { m_axi_B_V_RID ID 0 1 }  { m_axi_B_V_RUSER USER 0 1 }  { m_axi_B_V_RRESP RESP 0 2 }  { m_axi_B_V_BVALID VALID 0 1 }  { m_axi_B_V_BREADY READY 1 1 }  { m_axi_B_V_BRESP RESP 0 2 }  { m_axi_B_V_BID ID 0 1 }  { m_axi_B_V_BUSER USER 0 1 } } }
	B_V_offset { ap_fifo {  { B_V_offset_dout fifo_data 0 32 }  { B_V_offset_empty_n fifo_status 0 1 }  { B_V_offset_read fifo_update 1 1 } } }
	fifo_B_local_out_V_V { ap_fifo {  { fifo_B_local_out_V_V_din fifo_data 1 128 }  { fifo_B_local_out_V_V_full_n fifo_status 0 1 }  { fifo_B_local_out_V_V_write fifo_update 1 1 } } }
}
