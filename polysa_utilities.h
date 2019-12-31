#ifndef _POLYSA_UTILITIES_H
#define _POLYSA_UTILITIES_H

#include <isl/ast.h>
#include <isl/id.h>
#include <isl/id_to_ast_expr.h>

#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"

isl_stat unionize_pw_aff_space(__isl_take isl_pw_aff *pa, void *user);
__isl_give isl_union_map *extract_sizes_from_str(isl_ctx *ctx, const char *str);

isl_stat concat_basic_map(__isl_take isl_map *el, void *user);
__isl_give isl_basic_map_list *isl_union_map_get_basic_map_list(__isl_keep isl_union_map *umap);
isl_size isl_union_map_n_basic_map(__isl_keep isl_union_map *umap);
__isl_give isl_basic_map *isl_basic_map_from_map(__isl_take isl_map *map);

void print_mat(FILE *fp, __isl_keep isl_mat *mat);

#endif