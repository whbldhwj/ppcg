#ifndef _POLYSA_GROUP_H_
#define _POLYSA_GROUP_H_

#include "polysa_common.h"

int polysa_group_references(struct polysa_kernel *kernel,
  __isl_keep isl_schedule_node *node);
void polysa_array_ref_group_compute_tiling(struct polysa_array_ref_group *group);
__isl_give isl_printer *polysa_array_ref_group_print_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p);
__isl_give isl_union_map *polysa_array_ref_group_access_relation(
	struct polysa_array_ref_group *group, int read, int write);

#endif
