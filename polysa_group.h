#ifndef _POLYSA_GROUP_H_
#define _POLYSA_GROUP_H_

#include "polysa_common.h"

int polysa_group_references(struct polysa_kernel *kernel,
  __isl_keep isl_schedule_node *node);
isl_stat sa_group_references(struct polysa_kernel *kernel);
void polysa_array_ref_group_compute_tiling(struct polysa_array_tile *tile,
    struct polysa_array_ref_group *group);
__isl_give isl_printer *polysa_array_ref_group_print_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p);
__isl_give isl_union_map *polysa_array_ref_group_access_relation(
	struct polysa_array_ref_group *group, int read, int write);
__isl_give isl_union_map *polysa_io_group_access_relation(
  struct polysa_array_ref_group *group, int read, int write);
__isl_give isl_union_map *polysa_drain_group_access_relation(
  struct polysa_array_ref_group *group, int read, int write, 
  isl_union_set *domain);
__isl_give isl_union_map *polysa_io_group_ref_access_relation(
  struct polysa_array_ref_group *group,
  struct polysa_stmt_access *ref,
  int read, int write);
isl_bool can_tile(__isl_keep isl_map *access,
	struct polysa_array_tile *tile);
//void polysa_array_ref_reg_compute_tiling(
//  struct polysa_array_tile *tile,
//  struct polysa_stmt_access *access,
//  struct polysa_array_ref_group *group);
__isl_give isl_schedule *get_io_schedule(__isl_take isl_schedule *schedule, __isl_keep isl_vec *dir, __isl_give isl_multi_aff **io_trans, __isl_give isl_mat **io_trans_mat, int hbm);
__isl_give isl_printer *polysa_array_ref_group_print_fifo_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p);
__isl_give isl_multi_aff *polysa_array_ref_group_recompute_tiling(
  struct polysa_array_tile *tile,
  struct polysa_array_ref_group *group,
  int depth);

#endif
