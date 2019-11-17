#ifndef _POLYSA_COMMON_H_
#define _POLYSA_COMMON_H_

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "ppcg.h"

struct polysa_iter {
  char *name;
  isl_aff *lb;
  isl_aff *ub;
  int stride;
  char *ts_name;
};

struct polysa_prog {
  isl_ctx *ctx;
  isl_schedule *schedule;
  struct ppcg_scop *scop;

  int array_dim;
  int array_part_w;
  int space_w;
  int time_w;

  int type; // 0 - async 1 - sync 
};

struct polysa_acc {
  isl_map *tagged_map;
  isl_map *map;
  isl_space *id;

  int rw; // 0 - read 1 - write
};
void print_mat(FILE *fp, __isl_keep isl_mat *mat);
isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
struct polysa_prog **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop, isl_size *num_sa);
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule);
__isl_give isl_schedule_node *get_innermost_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_permutable_node_update(__isl_keep isl_schedule_node *node, void *user);
isl_stat sa_pe_optimize(struct polysa_prog *sa);
isl_stat sa_loop_init(struct polysa_prog *sa);
struct polysa_prog *sa_candidates_smart_pick(struct polysa_prog **sa_list, __isl_keep isl_size num_sa);
struct polysa_prog **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_prog **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_prog **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
__isl_give isl_schedule *loop_interchange_at_node(__isl_take isl_schedule_node *node, isl_size level1, isl_size level2);
isl_stat sa_latency_hiding_optimize(struct polysa_prog *sa);
isl_stat sa_SIMD_vectorization_optimize(struct polysa_prog *sa);
isl_stat sa_array_partitioning_optimize(struct polysa_prog *sa);
void *polysa_prog_free(struct polysa_prog *sa);
struct polysa_prog *polysa_prog_copy(struct polysa_prog *sa);
struct polysa_prog *polysa_prog_from_schedule(__isl_take isl_schedule *schedule);
void *polysa_acc_free(struct polysa_acc *acc);

/* Utils */
isl_size isl_union_map_n_basic_map(__isl_keep isl_union_map *umap);
__isl_give isl_basic_map_list *isl_union_map_get_basic_map_list(__isl_keep isl_union_map *umap);
__isl_give isl_basic_map *isl_basic_map_from_map(__isl_take isl_map *map);
__isl_give isl_vec *get_dep_dis_at_node(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule_node *band);
__isl_give isl_vec *get_dep_dis_at_schedule(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule *schedule);
__isl_null struct polysa_iter *polysa_iter_free(struct polysa_iter *iter);

#endif
