#ifndef _POLYSA_CPU_H
#define _POLYSA_CPU_H

#include <isl/ctx.h>

#include "ppcg.h"

struct ppcg_options;

//isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
//    struct ppcg_scop *ps, struct ppcg_options *options);
int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *ppcg_options,
    const char *input, const char *output);

#endif
