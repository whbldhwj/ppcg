#ifndef POLYSA_TREE_H
#define POLYSA_TREE_H

#include <isl/schedule_node.h>

__isl_give isl_schedule_node *polysa_tree_move_down_to_array(
  __isl_take isl_schedule_node *node, __isl_keep isl_union_set *core);
__isl_give isl_schedule_node *polysa_tree_move_up_to_array(
  __isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *polysa_tree_move_up_to_anchor(
  __isl_take isl_schedule_node *node);

#endif
