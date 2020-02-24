#ifndef _POLYSA_FPGA_H
#define _POLYSA_FPGA_H

#include <isl/ctx.h>
#include <isl/id_to_id.h>

#include "polysa_common.h"
#include "polysa_print.h"

int generate_polysa_xilinx_hls(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input);
int generate_polysa_intel_opencl(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input);

#endif
