#ifndef _POLYSA_SA_H
#define _POLYSA_SA_H

#include <isl/ast.h>
#include <isl/id.h>
#include <isl/id_to_ast_expr.h>

#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "polysa_common.h"

struct polysa_vsa {
  int array_part_w;
  int space_w;
  int time_w;

  int t2s_iter_num;
  char **t2s_iters;
};

struct polysa_vsa *polysa_vsa_alloc();
void *polysa_vsa_free(struct polysa_vsa *vsa);

void vsa_band_width_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);
void vsa_t2s_iter_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);
void vsa_t2s_var_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);

#endif
