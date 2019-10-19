#ifndef _POLYSA_CPU_H
#define _POLYSA_CPU_H

#include <isl/ctx.h>

#include "ppcg.h"
#include "polysa_common.h"

int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *ppcg_options,
    const char *input, const char *output);

#endif
