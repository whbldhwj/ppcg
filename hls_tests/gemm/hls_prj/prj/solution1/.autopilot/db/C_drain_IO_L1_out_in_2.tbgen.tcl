set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 15
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "6", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "6", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "6", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 15
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "6", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "6", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "6", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 15
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "6", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "6", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "6", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
	{ en uint 1 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
	{ en sc_in sc_logic 1 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }} , 
 	{ "name": "en", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
	en { ap_none {  { en in_data 0 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
	{ en uint 1 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
	{ en sc_in sc_logic 1 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }} , 
 	{ "name": "en", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
	en { ap_none {  { en in_data 0 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
	{ en uint 1 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
	{ en sc_in sc_logic 1 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }} , 
 	{ "name": "en", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
	en { ap_none {  { en in_data 0 1 } } }
}
set moduleName C_drain_IO_L1_out_in_2
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
set C_modelName {C_drain_IO_L1_out_in.2}
set C_modelType { void 0 }
set C_modelArgList {
	{ local_C_0_V int 64 regular {array 2 { 1 3 } 1 1 }  }
	{ fifo_C_drain_in_V_V int 64 regular {fifo 0 volatile }  }
	{ fifo_C_drain_out_V_V int 64 regular {fifo 1 volatile }  }
	{ en uint 1 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "local_C_0_V", "interface" : "memory", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_in_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_C_drain_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} , 
 	{ "Name" : "en", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ local_C_0_V_address0 sc_out sc_lv 1 signal 0 } 
	{ local_C_0_V_ce0 sc_out sc_logic 1 signal 0 } 
	{ local_C_0_V_q0 sc_in sc_lv 64 signal 0 } 
	{ fifo_C_drain_in_V_V_dout sc_in sc_lv 64 signal 1 } 
	{ fifo_C_drain_in_V_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ fifo_C_drain_in_V_V_read sc_out sc_logic 1 signal 1 } 
	{ fifo_C_drain_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_C_drain_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_C_drain_out_V_V_write sc_out sc_logic 1 signal 2 } 
	{ en sc_in sc_logic 1 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "local_C_0_V_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "address0" }} , 
 	{ "name": "local_C_0_V_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "ce0" }} , 
 	{ "name": "local_C_0_V_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "local_C_0_V", "role": "q0" }} , 
 	{ "name": "fifo_C_drain_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_C_drain_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_C_drain_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_C_drain_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_C_drain_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_C_drain_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_C_drain_out_V_V", "role": "write" }} , 
 	{ "name": "en", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "en", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "C_drain_IO_L1_out_in_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "1", "Max" : "6"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "6"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	local_C_0_V { ap_memory {  { local_C_0_V_address0 mem_address 1 1 }  { local_C_0_V_ce0 mem_ce 1 1 }  { local_C_0_V_q0 mem_dout 0 64 } } }
	fifo_C_drain_in_V_V { ap_fifo {  { fifo_C_drain_in_V_V_dout fifo_data 0 64 }  { fifo_C_drain_in_V_V_empty_n fifo_status 0 1 }  { fifo_C_drain_in_V_V_read fifo_update 1 1 } } }
	fifo_C_drain_out_V_V { ap_fifo {  { fifo_C_drain_out_V_V_din fifo_data 1 64 }  { fifo_C_drain_out_V_V_full_n fifo_status 0 1 }  { fifo_C_drain_out_V_V_write fifo_update 1 1 } } }
	en { ap_none {  { en in_data 0 1 } } }
}
