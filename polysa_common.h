#ifndef _POLYSA_COMMON_H_
#define _POLYSA_COMMON_H_

#include <assert.h>

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
isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
__isl_give isl_schedule **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop, isl_size *num_sa);
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_permutable_node_update(__isl_keep isl_schedule_node *node, void *user);
isl_stat sa_pe_optimize(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
__isl_give isl_schedule *sa_candidates_smart_pick(__isl_take isl_schedule **sa_list, struct ppcg_scop *scop, __isl_keep isl_size num_sa);
__isl_give isl_schedule **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
__isl_give isl_schedule **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
__isl_give isl_schedule **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);

#endif
