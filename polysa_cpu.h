#ifndef _POLYSA_CPU_H
#define _POLYSA_CPU_H

#include <isl/ctx.h>
#include <isl/id_to_id.h>

#include "polysa_common.h"
#include "polysa_print.h"

int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input);

#endif
