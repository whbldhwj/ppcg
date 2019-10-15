#ifndef _POLYSA_COMMON_H_
#define _POLYSA_COMMON_H_

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "ppcg.h"

void print_mat(FILE *fp, __isl_keep isl_mat *mat);
isl_bool is_permutable_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);

#endif
