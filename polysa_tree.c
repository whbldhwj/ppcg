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

/* Is "node" a mark node with an identifier called "local"?
 */
static int node_is_local(__isl_keep isl_schedule_node *node)
{
  return is_marked(node, "local");
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

/* Is "node" a mark node with an identifier called "kernel"?
 */
int polysa_tree_node_is_kernel(__isl_keep isl_schedule_node *node)
{
  return is_marked(node, "kernel");
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

/* Move down the branch between "kernel" and "arrray" until
 * the "local" mark is reached, where the branch containing the "local"
 * mark is identified by the domain elements in "core".
 */
__isl_give isl_schedule_node *polysa_tree_move_down_to_local(
  __isl_take isl_schedule_node *node, __isl_keep isl_union_set *core)
{
  int is_local;

  while ((is_local = node_is_local(node)) == 0)
    node = core_child(node, core);
  
  if (is_local < 0)
    node = isl_schedule_node_free(node);

  return node;
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

/* Move down from the "kernel" mark (or at least a node with schedule
 * depth smaller than or equal to "depth") to a band node at schedule
 * depth "depth".  The "array" mark is assumed to have a schedule
 * depth greater than or equal to "depth".  The branch containing the
 * "array" mark is identified by the domain elements in "core".
 *
 * If the desired schedule depth is in the middle of band node,
 * then the band node is split into two pieces, the second piece
 * at the desired schedule depth.
 */
__isl_give isl_schedule_node *polysa_tree_move_down_to_depth(
	__isl_take isl_schedule_node *node, int depth,
	__isl_keep isl_union_set *core)
{
  int is_local;
  int is_array = 0;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_set_get_ctx(core), stdout);
//  p = isl_printer_print_union_set(p, core);
//  printf("\n");
//  // debug

	while (node && isl_schedule_node_get_schedule_depth(node) < depth) {
		if (isl_schedule_node_get_type(node) ==
						    isl_schedule_node_band) {
			int node_depth, node_dim;
			node_depth = isl_schedule_node_get_schedule_depth(node);
			node_dim = isl_schedule_node_band_n_member(node);
			if (node_depth + node_dim > depth)
				node = isl_schedule_node_band_split(node,
							depth - node_depth);
		}
		node = core_child(node, core);
	}
	while ((is_local = node_is_local(node)) == 0 &&
	    (is_array = node_is_array(node)) == 0 &&
	    isl_schedule_node_get_type(node) != isl_schedule_node_band)
		node = core_child(node, core);
	if (is_local < 0 || is_array < 0)
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

/* Move up the tree underneath the "kernel" mark until
 * the "kernel" mark is reached.
 */
__isl_give isl_schedule_node *polysa_tree_move_up_to_kernel(
  __isl_take isl_schedule_node *node)
{
  int is_kernel;

  while ((is_kernel = polysa_tree_node_is_kernel(node)) == 0) {
    // debug
    isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
    p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
    p = isl_printer_print_schedule_node(p, node);
    printf("\n");
    // debug
    node = isl_schedule_node_parent(node);
  }
  if (is_kernel < 0)
    node = isl_schedule_node_free(node);

  return node;
}

/* Insert a mark node with identifier "local" in front of "node".
 */
static __isl_give isl_schedule_node *insert_local(
  __isl_take isl_schedule_node *node)
{
  isl_ctx *ctx;
  isl_id *id;

  ctx = isl_schedule_node_get_ctx(node);
  id = isl_id_alloc(ctx, "local", NULL);
  node = isl_schedule_node_insert_mark(node, id);

  return node;
}

/* Insert a "local" mark in front of the "array" mark 
 * provided the linear branch between "node" and the "array" mark
 * does not contain such a "local" mark already.
 *
 * As a side effect, this function checks that the subtree at "node"
 * actually contains a "array" mark and that there is no branching
 * in between "node" and this "array" mark.
 */
__isl_give isl_schedule_node *polysa_tree_insert_local_before_array(
  __isl_take isl_schedule_node *node)
{
  int depth0, depth;
  int any_local = 0;

  if (!node)
    return NULL;

  depth0 = isl_schedule_node_get_tree_depth(node);

  for (;;) {
    int is_array;
    int n;

    if (!any_local) {
      any_local = node_is_local(node); 
      if (any_local < 0)
        return isl_schedule_node_free(node);
    }
    is_array = node_is_array(node); 
    if (is_array < 0) 
      return isl_schedule_node_free(node);
    if (is_array)
      break;
    n = isl_schedule_node_n_children(node);
    if (n == 0)
      isl_die(isl_schedule_node_get_ctx(node),
          isl_error_invalid,
          "no array marker found",
          return isl_schedule_node_free(node));
    if (n > 1)
      isl_die(isl_schedule_node_get_ctx(node),
          isl_error_invalid,
          "expecting single array marker",
          return isl_schedule_node_free(node));

    node = isl_schedule_node_child(node, 0);
  }

  if (!any_local)
    node = insert_local(node);
  depth = isl_schedule_node_get_tree_depth(node);
  node = isl_schedule_node_ancestor(node, depth - depth0);

  return node;
}


