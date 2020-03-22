set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "137", "EstimateLatencyMax" : "202",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "87", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "86", "EstimateLatencyMax" : "184",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_138", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_124", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "137", "Max" : "202"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "185"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "137", "EstimateLatencyMax" : "202",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "87", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "86", "EstimateLatencyMax" : "184",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_138", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_124", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "137", "Max" : "202"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "185"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "137", "EstimateLatencyMax" : "195",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "87", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "86", "EstimateLatencyMax" : "176",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_138", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_138", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "168",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_124", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_124", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "76",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state8", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_129", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_129", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "137", "Max" : "195"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "177"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "135", "EstimateLatencyMax" : "147",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
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
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "135", "Max" : "147"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "121"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 31,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "135", "EstimateLatencyMax" : "147",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
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
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "135", "Max" : "147"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "121"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 31,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "135", "EstimateLatencyMax" : "147",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
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
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "135", "Max" : "147"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "121"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
set moduleName kernel0
set isTopModule 1
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {kernel0}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem_A int 128 regular {axi_master 0}  }
	{ gmem_B int 128 regular {axi_master 0}  }
	{ gmem_C int 64 regular {axi_master 1}  }
	{ A_V int 32 regular {axi_slave 0}  }
	{ B_V int 32 regular {axi_slave 0}  }
	{ C_V int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem_A", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "A.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "A_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_B", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[{"low":0,"up":127,"cElement": [{"cName": "B.V","cData": "uint128","bit_use": { "low": 0,"up": 127},"offset": { "type": "dynamic","port_name": "B_V","bundle": "control"},"direction": "READONLY","cArray": [{"low" : 0,"up" : 15,"step" : 1}]}]}]} , 
 	{ "Name" : "gmem_C", "interface" : "axi_master", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "C.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"offset": { "type": "dynamic","port_name": "C_V","bundle": "control"},"direction": "WRITEONLY","cArray": [{"low" : 0,"up" : 31,"step" : 1}]}]}]} , 
 	{ "Name" : "A_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "B_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "C_V", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
	{ m_axi_gmem_A_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_AWADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem_A_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_ARADDR sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem_A_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem_A_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem_A_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem_A_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem_A_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem_A_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem_A_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_A_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem_B_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_AWADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WDATA sc_out sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_WSTRB sc_out sc_lv 16 signal 1 } 
	{ m_axi_gmem_B_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_ARADDR sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem_B_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem_B_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem_B_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem_B_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RDATA sc_in sc_lv 128 signal 1 } 
	{ m_axi_gmem_B_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem_B_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem_B_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_B_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem_C_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_AWADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_ARADDR sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem_C_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem_C_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem_C_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem_C_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_gmem_C_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem_C_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem_C_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem_C_BUSER sc_in sc_lv 1 signal 2 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"continue","value":"0","valid_bit":"4"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"},{"name":"A_V","role":"data","value":"16"},{"name":"B_V","role":"data","value":"24"},{"name":"C_V","role":"data","value":"32"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"kernel0","role":"start","value":"0","valid_bit":"0"},{"name":"kernel0","role":"done","value":"0","valid_bit":"1"},{"name":"kernel0","role":"idle","value":"0","valid_bit":"2"},{"name":"kernel0","role":"ready","value":"0","valid_bit":"3"},{"name":"kernel0","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem_A_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_A_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_A_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_A_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_A_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_A_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_A_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_A_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_A_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_A_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_A_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_A_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_A_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_A_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_A_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_A_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_A_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_A", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_A_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_A_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_A_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_A_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_A_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_A_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_A_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_A_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_A_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_A_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_A_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_A_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_A_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_A_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_A_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_A_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_A_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_A_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_A_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_A", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_A_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_A_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_A_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_A_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_A_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_A_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_A_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_A", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_A_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_A_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_A", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_B_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_B_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_B_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_B_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_B_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_B_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_B_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_B_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_B_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_B_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_B_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_B_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_B_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_B_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_B_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_B_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_B_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem_B", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_B_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_B_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_B_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_B_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_B_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_B_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_B_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_B_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_B_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_B_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_B_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_B_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_B_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_B_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_B_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_B_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_B_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_B_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_B_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem_B", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_B_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_B_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_B_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_B_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_B_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_B_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_B_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_B", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_B_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_B_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_B", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem_C_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem_C_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem_C_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem_C_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem_C_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem_C_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem_C_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem_C_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem_C_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem_C_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem_C_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem_C_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem_C_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem_C_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem_C_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem_C_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem_C_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem_C_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem_C_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WID" }} , 
 	{ "name": "m_axi_gmem_C_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem_C_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem_C_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem_C_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem_C_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem_C_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem_C_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem_C_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem_C_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem_C_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem_C_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem_C_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem_C_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem_C_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem_C_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem_C_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem_C_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem_C", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem_C_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem_C_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RID" }} , 
 	{ "name": "m_axi_gmem_C_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem_C_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem_C_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem_C_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem_C_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem_C", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem_C_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BID" }} , 
 	{ "name": "m_axi_gmem_C_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem_C", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "12", "17", "18", "23", "28", "30", "32", "33", "35", "36", "38", "39", "40", "45", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"],
		"CDFG" : "kernel0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "135", "EstimateLatencyMax" : "147",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"InputProcess" : [
			{"ID" : "5", "Name" : "kernel0_entry6_U0", "ReadyCount" : "kernel0_entry6_U0_ap_ready_count"},
			{"ID" : "6", "Name" : "A_IO_L3_in_U0", "ReadyCount" : "A_IO_L3_in_U0_ap_ready_count"},
			{"ID" : "17", "Name" : "B_IO_L3_in_U0", "ReadyCount" : "B_IO_L3_in_U0_ap_ready_count"}],
		"OutputProcess" : [
			{"ID" : "32", "Name" : "PE_A_dummy141_U0"},
			{"ID" : "35", "Name" : "PE_B_dummy143_U0"},
			{"ID" : "38", "Name" : "PE_A_dummy_U0"},
			{"ID" : "39", "Name" : "PE_B_dummy_U0"},
			{"ID" : "62", "Name" : "C_drain_IO_L3_out_U0"}],
		"Port" : [
			{"Name" : "gmem_A", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "A_IO_L3_in_U0", "Port" : "A_V"}]},
			{"Name" : "gmem_B", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "17", "SubInstance" : "B_IO_L3_in_U0", "Port" : "B_V"}]},
			{"Name" : "gmem_C", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "62", "SubInstance" : "C_drain_IO_L3_out_U0", "Port" : "C_V"}]},
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_control_s_axi_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_A_m_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_B_m_axi_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_gmem_C_m_axi_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.kernel0_entry6_U0", "Parent" : "0",
		"CDFG" : "kernel0_entry6",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "A_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "B_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "C_V", "Type" : "None", "Direction" : "I"},
			{"Name" : "A_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "6", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "B_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "17", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_out_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_out", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L3_in_U0", "Parent" : "0",
		"CDFG" : "A_IO_L3_in",
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
			{"Name" : "A_V", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "A_V_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "A_V_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "A_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "63",
				"BlockSignal" : [
					{"Name" : "A_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "7", "DependentChan" : "66",
				"BlockSignal" : [
					{"Name" : "fifo_A_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0", "Parent" : "0", "Child" : ["8", "9", "10", "11"],
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
		"StartSource" : "6",
		"StartFifo" : "start_for_A_IO_L2sc4_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "6", "DependentChan" : "66",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "12", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "11", "SubInstance" : "grp_A_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_A_out_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "68",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_ping_0_V_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.local_A_pong_0_V_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_intra_tra_fu_131", "Parent" : "7",
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
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_U0.grp_A_IO_L2_in_inter_tra_1_fu_139", "Parent" : "7",
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
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["13", "14", "15", "16"],
		"CDFG" : "A_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_A_IO_L2tde_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "67",
				"SubConnect" : [
					{"ID" : "16", "SubInstance" : "grp_A_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_A_in_V_V"}]},
			{"Name" : "fifo_A_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "69",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_A_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_A_local_out_V_V"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_ping_0_V_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.local_A_pong_0_V_U", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_intra_tra_fu_117", "Parent" : "12",
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
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.A_IO_L2_in_tail_U0.grp_A_IO_L2_in_inter_tra_fu_125", "Parent" : "12",
		"CDFG" : "A_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
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
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L3_in_U0", "Parent" : "0",
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
			{"Name" : "B_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "64",
				"BlockSignal" : [
					{"Name" : "B_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "18", "DependentChan" : "70",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0", "Parent" : "0", "Child" : ["19", "20", "21", "22"],
		"CDFG" : "B_IO_L2_in",
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
		"StartSource" : "17",
		"StartFifo" : "start_for_B_IO_L2udo_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "17", "DependentChan" : "70",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "23", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_B_IO_L2_in_inter_tra_1_fu_139", "Port" : "fifo_B_out_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "28", "DependentChan" : "72",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_131", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_ping_0_V_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.local_B_pong_0_V_U", "Parent" : "18"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_intra_tra_fu_131", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_U0.grp_B_IO_L2_in_inter_tra_1_fu_139", "Parent" : "18",
		"CDFG" : "B_IO_L2_in_inter_tra_1",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0", "Parent" : "0", "Child" : ["24", "25", "26", "27"],
		"CDFG" : "B_IO_L2_in_tail",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "63", "EstimateLatencyMax" : "120",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "18",
		"StartFifo" : "start_for_B_IO_L2vdy_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state6", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125"}],
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "71",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_B_IO_L2_in_inter_tra_fu_125", "Port" : "fifo_B_in_V_V"}]},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "73",
				"SubConnect" : [
					{"ID" : "26", "SubInstance" : "grp_B_IO_L2_in_intra_tra_fu_117", "Port" : "fifo_B_local_out_V_V"}]}]},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_ping_0_V_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.local_B_pong_0_V_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_intra_tra_fu_117", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_intra_tra",
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
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_B_local_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_B_local_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.B_IO_L2_in_tail_U0.grp_B_IO_L2_in_inter_tra_fu_125", "Parent" : "23",
		"CDFG" : "B_IO_L2_in_inter_tra",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_B_0_V", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE139_U0", "Parent" : "0", "Child" : ["29"],
		"CDFG" : "PE139",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "7",
		"StartFifo" : "start_for_PE139_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "7", "DependentChan" : "68",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "30", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "18", "DependentChan" : "72",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "33", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "40", "DependentChan" : "76",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE139_U0.local_C_U", "Parent" : "28"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE140_U0", "Parent" : "0", "Child" : ["31"],
		"CDFG" : "PE140",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "23",
		"StartFifo" : "start_for_PE140_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "74",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "32", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "23", "DependentChan" : "73",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "50", "DependentChan" : "79",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE140_U0.local_C_U", "Parent" : "30"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy141_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy141",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_A_duxdS_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "77",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE142_U0", "Parent" : "0", "Child" : ["34"],
		"CDFG" : "PE142",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "12",
		"StartFifo" : "start_for_PE142_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "12", "DependentChan" : "69",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "36", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "75",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "35", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "82",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE142_U0.local_C_U", "Parent" : "33"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy143_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy143",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_PE_B_duzec_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "81",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_U0", "Parent" : "0", "Child" : ["37"],
		"CDFG" : "PE",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "109", "EstimateLatencyMax" : "109",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_PE_U0_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "80",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_A_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "38", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "78",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_B_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "39", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "85",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.PE_U0.local_C_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_A_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_A_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_A_duBew_U",
		"Port" : [
			{"Name" : "fifo_A_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "83",
				"BlockSignal" : [
					{"Name" : "fifo_A_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.PE_B_dummy_U0", "Parent" : "0",
		"CDFG" : "PE_B_dummy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_PE_B_duCeG_U",
		"Port" : [
			{"Name" : "fifo_B_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "84",
				"BlockSignal" : [
					{"Name" : "fifo_B_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0", "Parent" : "0", "Child" : ["41", "42", "43", "44"],
		"CDFG" : "C_drain_IO_L1_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "28",
		"StartFifo" : "start_for_C_drainwdI_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "45", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "44", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "28", "DependentChan" : "76",
				"SubConnect" : [
					{"ID" : "43", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_ping_0_V_U", "Parent" : "40"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.local_C_pong_0_V_U", "Parent" : "40"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "40",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0", "Parent" : "0", "Child" : ["46", "47", "48", "49"],
		"CDFG" : "C_drain_IO_L1_out145",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "33",
		"StartFifo" : "start_for_C_drainAem_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "40", "DependentChan" : "86",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "60", "DependentChan" : "87",
				"SubConnect" : [
					{"ID" : "48", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "33", "DependentChan" : "82",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_ping_0_V_U", "Parent" : "45"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.local_C_pong_0_V_U", "Parent" : "45"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "45",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out145_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "45",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0", "Parent" : "0", "Child" : ["51", "52", "53", "54"],
		"CDFG" : "C_drain_IO_L1_out_he_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "30",
		"StartFifo" : "start_for_C_drainyd2_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106"},
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113"}],
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "55", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_C_drain_IO_L1_out_in_1_fu_113", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "30", "DependentChan" : "79",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_106", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_ping_0_V_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.local_C_pong_0_V_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_fu_106", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_he_1_U0.grp_C_drain_IO_L1_out_in_1_fu_113", "Parent" : "50",
		"CDFG" : "C_drain_IO_L1_out_in_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "4",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0", "Parent" : "0", "Child" : ["56", "57", "58", "59"],
		"CDFG" : "C_drain_IO_L1_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39", "EstimateLatencyMax" : "44",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "36",
		"StartFifo" : "start_for_C_drainDeQ_U",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"},
			{"State" : "ap_ST_fsm_state4", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130"}],
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "50", "DependentChan" : "88",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_in_V_V"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "89",
				"SubConnect" : [
					{"ID" : "58", "SubInstance" : "grp_C_drain_IO_L1_out_in_2_fu_120", "Port" : "fifo_C_drain_out_V_V"}]},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "36", "DependentChan" : "85",
				"SubConnect" : [
					{"ID" : "59", "SubInstance" : "grp_C_drain_IO_L1_out_in_fu_130", "Port" : "fifo_C_drain_local_in_V"}]}]},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_ping_0_V_U", "Parent" : "55"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.local_C_pong_0_V_U", "Parent" : "55"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_2_fu_120", "Parent" : "55",
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
			{"Name" : "en", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L1_out_U0.grp_C_drain_IO_L1_out_in_fu_130", "Parent" : "55",
		"CDFG" : "C_drain_IO_L1_out_in",
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
			{"Name" : "local_C_0_V", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "fifo_C_drain_local_in_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_he_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out_he",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "18", "EstimateLatencyMax" : "18",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "45",
		"StartFifo" : "start_for_C_drainEe0_U",
		"Port" : [
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "61", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "45", "DependentChan" : "87",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L2_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L2_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "55",
		"StartFifo" : "start_for_C_drainFfa_U",
		"Port" : [
			{"Name" : "fifo_C_drain_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "60", "DependentChan" : "90",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_in_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_out_V_V", "Type" : "Fifo", "Direction" : "O", "DependentProc" : "62", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_out_V_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "55", "DependentChan" : "89",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_drain_IO_L3_out_U0", "Parent" : "0",
		"CDFG" : "C_drain_IO_L3_out",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"StartSource" : "5",
		"StartFifo" : "start_for_C_drainrcU_U",
		"Port" : [
			{"Name" : "C_V", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "C_V_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "C_V_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "C_V_offset", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "5", "DependentChan" : "65",
				"BlockSignal" : [
					{"Name" : "C_V_offset_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "fifo_C_drain_local_in_V_V", "Type" : "Fifo", "Direction" : "I", "DependentProc" : "61", "DependentChan" : "91",
				"BlockSignal" : [
					{"Name" : "fifo_C_drain_local_in_V_V_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.A_V_c_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.B_V_c_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.C_V_c_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_A_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "70", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_0_s_U", "Parent" : "0"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_B_IO_L2_in_1_s_U", "Parent" : "0"},
	{"ID" : "72", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_0_V_V_U", "Parent" : "0"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_1_V_V_U", "Parent" : "0"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_0_V_V_U", "Parent" : "0"},
	{"ID" : "76", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_0_s_U", "Parent" : "0"},
	{"ID" : "77", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_0_2_V_V_U", "Parent" : "0"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_0_1_s_U", "Parent" : "0"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_1_V_V_U", "Parent" : "0"},
	{"ID" : "81", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_0_V_V_U", "Parent" : "0"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_0_s_U", "Parent" : "0"},
	{"ID" : "83", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_A_PE_1_2_V_V_U", "Parent" : "0"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_B_PE_2_1_V_V_U", "Parent" : "0"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_PE_1_1_s_U", "Parent" : "0"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_6_U", "Parent" : "0"},
	{"ID" : "87", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_7_U", "Parent" : "0"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_8_U", "Parent" : "0"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_9_U", "Parent" : "0"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_10_U", "Parent" : "0"},
	{"ID" : "91", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fifo_C_drain_C_drain_11_U", "Parent" : "0"},
	{"ID" : "92", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainrcU_U", "Parent" : "0"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2sc4_U", "Parent" : "0"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_A_IO_L2tde_U", "Parent" : "0"},
	{"ID" : "95", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE139_U0_U", "Parent" : "0"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE142_U0_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2udo_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_B_IO_L2vdy_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE140_U0_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainwdI_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duxdS_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_U0_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainyd2_U", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duzec_U", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainAem_U", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_A_duBew_U", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_PE_B_duCeG_U", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainDeQ_U", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainEe0_U", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_C_drainFfa_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	kernel0 {
		gmem_A {Type I LastRead 9 FirstWrite -1}
		gmem_B {Type I LastRead 9 FirstWrite -1}
		gmem_C {Type O LastRead 4 FirstWrite 3}
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}}
	kernel0_entry6 {
		A_V {Type I LastRead 0 FirstWrite -1}
		B_V {Type I LastRead 0 FirstWrite -1}
		C_V {Type I LastRead 0 FirstWrite -1}
		A_V_out {Type O LastRead -1 FirstWrite 0}
		B_V_out {Type O LastRead -1 FirstWrite 0}
		C_V_out {Type O LastRead -1 FirstWrite 0}}
	A_IO_L3_in {
		A_V {Type I LastRead 9 FirstWrite -1}
		A_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
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
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_tail {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	A_IO_L2_in_intra_tra {
		local_A_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_A_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	A_IO_L2_in_inter_tra {
		local_A_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	B_IO_L3_in {
		B_V {Type I LastRead 9 FirstWrite -1}
		B_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 10}}
	B_IO_L2_in {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra_1 {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_tail {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}}
	B_IO_L2_in_intra_tra {
		local_B_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_B_local_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	B_IO_L2_in_inter_tra {
		local_B_0_V {Type O LastRead -1 FirstWrite 2}
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE139 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE140 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy141 {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE142 {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_B_dummy143 {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE {
		fifo_A_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_A_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_B_in_V_V {Type I LastRead 5 FirstWrite -1}
		fifo_B_out_V_V {Type O LastRead -1 FirstWrite 5}
		fifo_C_drain_out_V {Type O LastRead -1 FirstWrite 7}}
	PE_A_dummy {
		fifo_A_in_V_V {Type I LastRead 2 FirstWrite -1}}
	PE_B_dummy {
		fifo_B_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out145 {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_he_1 {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_1 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L1_out_in_2 {
		local_C_0_V {Type I LastRead 1 FirstWrite -1}
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		en {Type I LastRead 0 FirstWrite -1}}
	C_drain_IO_L1_out_in {
		local_C_0_V {Type IO LastRead 1 FirstWrite 2}
		fifo_C_drain_local_in_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out_he {
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L2_out {
		fifo_C_drain_in_V_V {Type I LastRead 2 FirstWrite -1}
		fifo_C_drain_out_V_V {Type O LastRead -1 FirstWrite 2}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}
	C_drain_IO_L3_out {
		C_V {Type O LastRead 4 FirstWrite 3}
		C_V_offset {Type I LastRead 0 FirstWrite -1}
		fifo_C_drain_local_in_V_V {Type I LastRead 2 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "135", "Max" : "147"}
	, {"Name" : "Interval", "Min" : "110", "Max" : "121"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem_A { m_axi {  { m_axi_gmem_A_AWVALID VALID 1 1 }  { m_axi_gmem_A_AWREADY READY 0 1 }  { m_axi_gmem_A_AWADDR ADDR 1 32 }  { m_axi_gmem_A_AWID ID 1 1 }  { m_axi_gmem_A_AWLEN LEN 1 8 }  { m_axi_gmem_A_AWSIZE SIZE 1 3 }  { m_axi_gmem_A_AWBURST BURST 1 2 }  { m_axi_gmem_A_AWLOCK LOCK 1 2 }  { m_axi_gmem_A_AWCACHE CACHE 1 4 }  { m_axi_gmem_A_AWPROT PROT 1 3 }  { m_axi_gmem_A_AWQOS QOS 1 4 }  { m_axi_gmem_A_AWREGION REGION 1 4 }  { m_axi_gmem_A_AWUSER USER 1 1 }  { m_axi_gmem_A_WVALID VALID 1 1 }  { m_axi_gmem_A_WREADY READY 0 1 }  { m_axi_gmem_A_WDATA DATA 1 128 }  { m_axi_gmem_A_WSTRB STRB 1 16 }  { m_axi_gmem_A_WLAST LAST 1 1 }  { m_axi_gmem_A_WID ID 1 1 }  { m_axi_gmem_A_WUSER USER 1 1 }  { m_axi_gmem_A_ARVALID VALID 1 1 }  { m_axi_gmem_A_ARREADY READY 0 1 }  { m_axi_gmem_A_ARADDR ADDR 1 32 }  { m_axi_gmem_A_ARID ID 1 1 }  { m_axi_gmem_A_ARLEN LEN 1 8 }  { m_axi_gmem_A_ARSIZE SIZE 1 3 }  { m_axi_gmem_A_ARBURST BURST 1 2 }  { m_axi_gmem_A_ARLOCK LOCK 1 2 }  { m_axi_gmem_A_ARCACHE CACHE 1 4 }  { m_axi_gmem_A_ARPROT PROT 1 3 }  { m_axi_gmem_A_ARQOS QOS 1 4 }  { m_axi_gmem_A_ARREGION REGION 1 4 }  { m_axi_gmem_A_ARUSER USER 1 1 }  { m_axi_gmem_A_RVALID VALID 0 1 }  { m_axi_gmem_A_RREADY READY 1 1 }  { m_axi_gmem_A_RDATA DATA 0 128 }  { m_axi_gmem_A_RLAST LAST 0 1 }  { m_axi_gmem_A_RID ID 0 1 }  { m_axi_gmem_A_RUSER USER 0 1 }  { m_axi_gmem_A_RRESP RESP 0 2 }  { m_axi_gmem_A_BVALID VALID 0 1 }  { m_axi_gmem_A_BREADY READY 1 1 }  { m_axi_gmem_A_BRESP RESP 0 2 }  { m_axi_gmem_A_BID ID 0 1 }  { m_axi_gmem_A_BUSER USER 0 1 } } }
	gmem_B { m_axi {  { m_axi_gmem_B_AWVALID VALID 1 1 }  { m_axi_gmem_B_AWREADY READY 0 1 }  { m_axi_gmem_B_AWADDR ADDR 1 32 }  { m_axi_gmem_B_AWID ID 1 1 }  { m_axi_gmem_B_AWLEN LEN 1 8 }  { m_axi_gmem_B_AWSIZE SIZE 1 3 }  { m_axi_gmem_B_AWBURST BURST 1 2 }  { m_axi_gmem_B_AWLOCK LOCK 1 2 }  { m_axi_gmem_B_AWCACHE CACHE 1 4 }  { m_axi_gmem_B_AWPROT PROT 1 3 }  { m_axi_gmem_B_AWQOS QOS 1 4 }  { m_axi_gmem_B_AWREGION REGION 1 4 }  { m_axi_gmem_B_AWUSER USER 1 1 }  { m_axi_gmem_B_WVALID VALID 1 1 }  { m_axi_gmem_B_WREADY READY 0 1 }  { m_axi_gmem_B_WDATA DATA 1 128 }  { m_axi_gmem_B_WSTRB STRB 1 16 }  { m_axi_gmem_B_WLAST LAST 1 1 }  { m_axi_gmem_B_WID ID 1 1 }  { m_axi_gmem_B_WUSER USER 1 1 }  { m_axi_gmem_B_ARVALID VALID 1 1 }  { m_axi_gmem_B_ARREADY READY 0 1 }  { m_axi_gmem_B_ARADDR ADDR 1 32 }  { m_axi_gmem_B_ARID ID 1 1 }  { m_axi_gmem_B_ARLEN LEN 1 8 }  { m_axi_gmem_B_ARSIZE SIZE 1 3 }  { m_axi_gmem_B_ARBURST BURST 1 2 }  { m_axi_gmem_B_ARLOCK LOCK 1 2 }  { m_axi_gmem_B_ARCACHE CACHE 1 4 }  { m_axi_gmem_B_ARPROT PROT 1 3 }  { m_axi_gmem_B_ARQOS QOS 1 4 }  { m_axi_gmem_B_ARREGION REGION 1 4 }  { m_axi_gmem_B_ARUSER USER 1 1 }  { m_axi_gmem_B_RVALID VALID 0 1 }  { m_axi_gmem_B_RREADY READY 1 1 }  { m_axi_gmem_B_RDATA DATA 0 128 }  { m_axi_gmem_B_RLAST LAST 0 1 }  { m_axi_gmem_B_RID ID 0 1 }  { m_axi_gmem_B_RUSER USER 0 1 }  { m_axi_gmem_B_RRESP RESP 0 2 }  { m_axi_gmem_B_BVALID VALID 0 1 }  { m_axi_gmem_B_BREADY READY 1 1 }  { m_axi_gmem_B_BRESP RESP 0 2 }  { m_axi_gmem_B_BID ID 0 1 }  { m_axi_gmem_B_BUSER USER 0 1 } } }
	gmem_C { m_axi {  { m_axi_gmem_C_AWVALID VALID 1 1 }  { m_axi_gmem_C_AWREADY READY 0 1 }  { m_axi_gmem_C_AWADDR ADDR 1 32 }  { m_axi_gmem_C_AWID ID 1 1 }  { m_axi_gmem_C_AWLEN LEN 1 8 }  { m_axi_gmem_C_AWSIZE SIZE 1 3 }  { m_axi_gmem_C_AWBURST BURST 1 2 }  { m_axi_gmem_C_AWLOCK LOCK 1 2 }  { m_axi_gmem_C_AWCACHE CACHE 1 4 }  { m_axi_gmem_C_AWPROT PROT 1 3 }  { m_axi_gmem_C_AWQOS QOS 1 4 }  { m_axi_gmem_C_AWREGION REGION 1 4 }  { m_axi_gmem_C_AWUSER USER 1 1 }  { m_axi_gmem_C_WVALID VALID 1 1 }  { m_axi_gmem_C_WREADY READY 0 1 }  { m_axi_gmem_C_WDATA DATA 1 64 }  { m_axi_gmem_C_WSTRB STRB 1 8 }  { m_axi_gmem_C_WLAST LAST 1 1 }  { m_axi_gmem_C_WID ID 1 1 }  { m_axi_gmem_C_WUSER USER 1 1 }  { m_axi_gmem_C_ARVALID VALID 1 1 }  { m_axi_gmem_C_ARREADY READY 0 1 }  { m_axi_gmem_C_ARADDR ADDR 1 32 }  { m_axi_gmem_C_ARID ID 1 1 }  { m_axi_gmem_C_ARLEN LEN 1 8 }  { m_axi_gmem_C_ARSIZE SIZE 1 3 }  { m_axi_gmem_C_ARBURST BURST 1 2 }  { m_axi_gmem_C_ARLOCK LOCK 1 2 }  { m_axi_gmem_C_ARCACHE CACHE 1 4 }  { m_axi_gmem_C_ARPROT PROT 1 3 }  { m_axi_gmem_C_ARQOS QOS 1 4 }  { m_axi_gmem_C_ARREGION REGION 1 4 }  { m_axi_gmem_C_ARUSER USER 1 1 }  { m_axi_gmem_C_RVALID VALID 0 1 }  { m_axi_gmem_C_RREADY READY 1 1 }  { m_axi_gmem_C_RDATA DATA 0 64 }  { m_axi_gmem_C_RLAST LAST 0 1 }  { m_axi_gmem_C_RID ID 0 1 }  { m_axi_gmem_C_RUSER USER 0 1 }  { m_axi_gmem_C_RRESP RESP 0 2 }  { m_axi_gmem_C_BVALID VALID 0 1 }  { m_axi_gmem_C_BREADY READY 1 1 }  { m_axi_gmem_C_BRESP RESP 0 2 }  { m_axi_gmem_C_BID ID 0 1 }  { m_axi_gmem_C_BUSER USER 0 1 } } }
}

set busDeadlockParameterList { 
	{ gmem_A { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_B { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
	{ gmem_C { NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 } } \
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem_A 1 }
	{ gmem_B 1 }
	{ gmem_C 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
