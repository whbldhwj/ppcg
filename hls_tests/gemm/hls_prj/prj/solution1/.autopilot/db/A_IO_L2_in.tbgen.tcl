set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "103", "EstimateLatencyMax" : "184",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "103", "Max" : "184"}
	, {"Name" : "Interval", "Min" : "103", "Max" : "184"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
set moduleName A_IO_L2_in
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
set C_modelName {A_IO_L2_in}
set C_modelType { void 0 }
set C_modelArgList {
	{ fifo_A_in_V_V int 128 regular {fifo 0 volatile }  }
	{ fifo_A_out_V_V int 128 regular {fifo 1 volatile }  }
	{ fifo_A_local_out_V_V int 64 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "fifo_A_in_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "READONLY"} , 
 	{ "Name" : "fifo_A_out_V_V", "interface" : "fifo", "bitwidth" : 128, "direction" : "WRITEONLY"} , 
 	{ "Name" : "fifo_A_local_out_V_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 19
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
	{ fifo_A_in_V_V_dout sc_in sc_lv 128 signal 0 } 
	{ fifo_A_in_V_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ fifo_A_in_V_V_read sc_out sc_logic 1 signal 0 } 
	{ fifo_A_out_V_V_din sc_out sc_lv 128 signal 1 } 
	{ fifo_A_out_V_V_full_n sc_in sc_logic 1 signal 1 } 
	{ fifo_A_out_V_V_write sc_out sc_logic 1 signal 1 } 
	{ fifo_A_local_out_V_V_din sc_out sc_lv 64 signal 2 } 
	{ fifo_A_local_out_V_V_full_n sc_in sc_logic 1 signal 2 } 
	{ fifo_A_local_out_V_V_write sc_out sc_logic 1 signal 2 } 
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
 	{ "name": "fifo_A_in_V_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "dout" }} , 
 	{ "name": "fifo_A_in_V_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "empty_n" }} , 
 	{ "name": "fifo_A_in_V_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_in_V_V", "role": "read" }} , 
 	{ "name": "fifo_A_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_out_V_V", "role": "write" }} , 
 	{ "name": "fifo_A_local_out_V_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "din" }} , 
 	{ "name": "fifo_A_local_out_V_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "full_n" }} , 
 	{ "name": "fifo_A_local_out_V_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "fifo_A_local_out_V_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "A_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "0", "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_ping_0_V_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.local_A_pong_0_V_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "0",
		"CDFG" : "A_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_A_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	A_IO_L2_in {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra_1 {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "120"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "120"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	fifo_A_in_V_V { ap_fifo {  { fifo_A_in_V_V_dout fifo_data 0 128 }  { fifo_A_in_V_V_empty_n fifo_status 0 1 }  { fifo_A_in_V_V_read fifo_update 1 1 } } }
	fifo_A_out_V_V { ap_fifo {  { fifo_A_out_V_V_din fifo_data 1 128 }  { fifo_A_out_V_V_full_n fifo_status 0 1 }  { fifo_A_out_V_V_write fifo_update 1 1 } } }
	fifo_A_local_out_V_V { ap_fifo {  { fifo_A_local_out_V_V_din fifo_data 1 64 }  { fifo_A_local_out_V_V_full_n fifo_status 0 1 }  { fifo_A_local_out_V_V_write fifo_update 1 1 } } }
}
