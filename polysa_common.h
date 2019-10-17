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

struct polysa_sa {
  isl_schedule *schedule;

  /* array dimension */
  isl_size array_dim;

  /* band width */
  isl_size array_part_w;
  isl_size space_w;
  isl_size time_w;
};

void print_mat(FILE *fp, __isl_keep isl_mat *mat);
isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
struct polysa_sa **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop, isl_size *num_sa);
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_permutable_node_update(__isl_keep isl_schedule_node *node, void *user);
isl_stat sa_pe_optimize(struct polysa_sa *sa, struct ppcg_scop *scop);
struct polysa_sa *sa_candidates_smart_pick(struct polysa_sa **sa_list, struct ppcg_scop *scop, __isl_keep isl_size num_sa);
struct polysa_sa **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_sa **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_sa **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
__isl_give isl_schedule *loop_interchange_at_node(__isl_take isl_schedule_node *node, isl_size level1, isl_size level2);

void polysa_sa_free(struct polysa_sa *sa);
struct polysa_sa *polysa_sa_copy(struct polysa_sa *sa);
struct polysa_sa *polysa_sa_from_schedule(__isl_take isl_schedule *schedule);

#endif
