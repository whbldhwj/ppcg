#ifndef _POLYSA_T2S_H
#define _POLYSA_T2S_H

#include <pet.h>
#include "ppcg_options.h"
#include "ppcg.h"
#include "polysa_common.h"
#include "string.h"

struct iter_exp {
  char *iter_name;
  int iter_offset;
};
typedef struct iter_exp IterExp;

struct acc_var_pair {
  isl_map *acc;
  char *var_name;
  char *var_ref;
  IterExp **var_iters;
  bool ei; // 0: external 1: intermediate
  bool d;  // 0: not drain 1: drain
};

int generate_polysa_t2s(isl_ctx *ctx, struct ppcg_options *options,
	const char *input);

#endif
