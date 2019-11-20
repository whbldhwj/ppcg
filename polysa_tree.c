#include <string.h>

#include <isl/space.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/id.h>

#include "polysa_tree.h"

/* Is "node" a mark node with an identifier called "name"?
 */
static int is_marked(__isl_keep isl_schedule_node *node, const char *name)
{
  isl_id *mark;
  int has_name;

  if (!node)
    return -1;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_mark)
    return 0;

  mark = isl_schedule_node_mark_get_id(node);
  if (!mark)
    return -1;

  has_name = !strcmp(isl_id_get_name(mark), name);
  isl_id_free(mark);

  return has_name;
}

static int node_is_array(__isl_keep isl_schedule_node *node)
{
  return is_marked(node, "array");
}

static int node_is_anchor(__isl_keep isl_schedule_node *node)
{
  return is_marked(node, "anchor");
}

/* Assuming "node" is a filter node, does it correspond to the branch
 * that contains the "array" mark, i.e., does it contain any elements in
 * "core"?
 */
static int node_is_core(__isl_keep isl_schedule_node *node,
  __isl_keep isl_union_set *core)
{
  int disjoint;
  isl_union_set *filter;

  filter = isl_schedule_node_filter_get_filter(node);
  disjoint = isl_union_set_is_disjoint(filter, core);
  isl_union_set_free(filter);
  if (disjoint < 0)
    return - 1;

  return !disjoint;
}

/* Move to the only child of "node" where the branch containing 
 * the domain elements in "core".
 *
 * If "node" is not a sequence, then it only has one child and we move
 * to that single child.
 * Otherwise, we check each of the filters in the children, pick
 * the one that corresponds to "core" and return a pointer to the child
 * of the filter node.
 */
static __isl_give isl_schedule_node *core_child(
  __isl_take isl_schedule_node *node, __isl_keep isl_union_set *core)
{
  int i, n;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
    return isl_schedule_node_child(node, 0);

  n = isl_schedule_node_n_children(node);
  for (i = 0; i < n; ++i) {
    int is_core;

    node = isl_schedule_node_child(node, i);
    is_core = node_is_core(node, core);

    if (is_core < 0)
      return isl_schedule_node_free(node);
    if (is_core)
      return isl_schedule_node_child(node, 0);

    node = isl_schedule_node_parent(node);
  }

  isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
    "core child not found", return isl_schedule_node_free(node));
}

/* Move down the branch until the "array" mark is reached,
 * where the branch containing the "array" mark is 
 * identified by the domain elements in "core".
 */
__isl_give isl_schedule_node *polysa_tree_move_down_to_array(
  __isl_take isl_schedule_node *node, __isl_keep isl_union_set *core)
{
  int is_array;

  while ((is_array = node_is_array(node)) == 0)
    node = core_child(node, core);

  if (is_array < 0)
    node = isl_schedule_node_free(node);

  return node;
}

/* Move up the tree underneath the "array" mark until the "array" mark is reached. 
 */
__isl_give isl_schedule_node *polysa_tree_move_up_to_array(
  __isl_take isl_schedule_node *node) {
  int is_array;

  while ((is_array = node_is_array(node)) == 0)
    node = isl_schedule_node_parent(node);
  
  if (is_array < 0)
    node = isl_schedule_node_free(node);

  return node;
}

/* Move up the tree underneath the "anchor" mark until the "anchor" mark is reached. 
 */
__isl_give isl_schedule_node *polysa_tree_move_up_to_anchor(
  __isl_take isl_schedule_node *node) {
  int is_anchor;

  while ((is_anchor = node_is_anchor(node)) == 0)
    node = isl_schedule_node_parent(node);
  
  if (is_anchor < 0)
    node = isl_schedule_node_free(node);

  return node;
}
