############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project prj
set_top kernel0
add_files src/kernel_xilinx.cpp
add_files src/kernel_kernel.h
add_files -tb src/kernel.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xcu200-fsgd2104-2-e}
create_clock -period 5 -name default
#source "./prj/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
