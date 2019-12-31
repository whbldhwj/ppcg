#ifndef _POLYSA_HLS_H
#define _POLYSA_HLS_H

#include <isl/ctx.h>
#include <isl/id_to_id.h>

#include "polysa_common.h"
#include "polysa_trans.h"
#include "polysa_print.h"
#include "polysa_codegen.h"

int generate_polysa_xilinx_hls(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input);

#endif
