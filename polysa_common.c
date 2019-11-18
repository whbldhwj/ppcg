#include "polysa_common.h"

static isl_stat concat_basic_map(__isl_take isl_map *el, void *user) 
{
  isl_basic_map_list **bmap_list = (isl_basic_map_list **)(user);
  isl_basic_map_list *bmap_list_sub = isl_map_get_basic_map_list(el);
  if (!(*bmap_list)) {
    *bmap_list = bmap_list_sub;
  } else {
    *bmap_list = isl_basic_map_list_concat(*bmap_list, bmap_list_sub);
  }

  isl_map_free(el);
  return isl_stat_ok;
}

__isl_give isl_basic_map_list *isl_union_map_get_basic_map_list(__isl_keep isl_union_map *umap)
{
  isl_map_list *map_list = isl_union_map_get_map_list(umap);
  isl_basic_map_list *bmap_list = NULL;
  isl_map_list_foreach(map_list, &concat_basic_map, &bmap_list);

  isl_map_list_free(map_list);
  return bmap_list;
}

static isl_stat acc_n_basic_map(__isl_take isl_map *el, void *user)
{
  isl_size *n = (isl_size *)(user);
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(el);
  *n = *n + isl_basic_map_list_n_basic_map(bmap_list);

//  // debug
//  isl_printer *printer = isl_printer_to_file(isl_map_get_ctx(el), stdout);
//  isl_printer_print_map(printer, el);
//  printf("\n");  
//  printf("%d\n", *n);
//  // debug

  isl_map_free(el);
  isl_basic_map_list_free(bmap_list);
  return isl_stat_ok;
}

isl_size isl_union_map_n_basic_map(__isl_keep isl_union_map *umap)
{
  isl_size n = 0;
  isl_map_list *map_list = isl_union_map_get_map_list(umap);
  isl_map_list_foreach(map_list, &acc_n_basic_map, &n);

  isl_map_list_free(map_list);

  return n;
}

__isl_give isl_basic_map *isl_basic_map_from_map(__isl_take isl_map *map)
{
  if (!map)
    return NULL;

  assert(isl_map_n_basic_map(map) == 1);
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  isl_map_free(map);

  isl_basic_map *bmap = isl_basic_map_list_get_basic_map(bmap_list, 0);
  isl_basic_map_list_free(bmap_list);

  return bmap;
}

/* Examines if the node is a permutable band node. */
static isl_bool is_permutable_node(__isl_keep isl_schedule_node *node) 
{
  if (!node)
    return isl_bool_error;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return isl_bool_false;
  if (!isl_schedule_node_band_get_permutable(node))
    return isl_bool_false;
  if (isl_schedule_node_band_n_member(node) < 1)
    return isl_bool_false;

  return isl_bool_true;
}

/* Examines if the node is a permutable band node. If so,
 * since the schedule tree is visited top-down,
 * return such a node immediately.
 */
isl_bool is_outermost_permutable_node_update(__isl_keep isl_schedule_node *node, void *user)
{
  isl_schedule_node **t_node = (isl_schedule_node **)(user);
  if (!node)
    return isl_bool_error;

  if (is_permutable_node(node) == isl_bool_true) {
    *t_node = isl_schedule_node_copy(node);
    return isl_bool_false;
  } else {
    return isl_bool_true;
  }

  return isl_bool_true;
}

static isl_bool no_permutable_node(isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band)
    return isl_bool_false;
  else
    return isl_bool_true;
}

/* Examines if the node is a permutable band node. If so,
 * since the schedule tree is visited bottom-up,
 * return the node immediately.
 */
isl_bool is_innermost_permutable_node_update(__isl_keep isl_schedule_node *node, void *user)
{
  isl_schedule_node **t_node = (isl_schedule_node **)(user);
  if (!node)
    return isl_bool_error;

  if (is_permutable_node(node) == isl_bool_true) {
    /* Check if there is any other band below it. */
    isl_schedule_node *new_node = isl_schedule_node_get_child(node, 0);
    isl_bool no_inner_band = isl_schedule_node_every_descendant(new_node,
        &no_permutable_node, NULL);
    if (no_inner_band) {
      if (*t_node == NULL)
        *t_node = isl_schedule_node_copy(node);
    }
    isl_schedule_node_free(new_node);
  }
  
  return isl_bool_true;
}

/* Examines if the node is a permutable band node. If so, 
 * increase the number of permutable node.
 */
isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user) {
  isl_val *n_permutable_node = (isl_val *)(user);
  if (!node)
    return isl_bool_error;

  if (is_permutable_node(node) == isl_bool_true)
    n_permutable_node = isl_val_add_ui(n_permutable_node, 1);

  return isl_bool_true;
}

/* Examines that if the program only contains one permutable node and there is
 * no other node beside it.
 */
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root;
  root = isl_schedule_get_root(schedule);
//  // debug
//  printf("%d\n", isl_schedule_node_get_type(root));
//  isl_printer *printer = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  isl_printer_print_schedule_node(printer, root);
//  printf("\n");
//  // debug
  isl_val *n_permutable_node = isl_val_zero(isl_schedule_get_ctx(schedule));
  isl_bool all_permutable_node = isl_schedule_node_every_descendant(root,
      &is_permutable_node_cnt, n_permutable_node);
  isl_schedule_node_free(root);
  if (all_permutable_node && isl_val_is_one(n_permutable_node)) {
    isl_val_free(n_permutable_node);
    return isl_bool_true;
  } else {
    isl_val_free(n_permutable_node);
    return isl_bool_false;
  }
}

/* Examines if the dependence is a uniform dependence based on the partial schedule
 * in the node.
 * We will calculate the dependence vector and examine if each dimension is a constant.
 */
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user)
{
  isl_basic_map *dep = (isl_basic_map *)(user);
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return isl_bool_true;

  /* By this stage we know that if a node is a band node, it is then a 
   * permutable band node to be analyzed. 
   */
  isl_multi_union_pw_aff *p_sc = isl_schedule_node_band_get_partial_schedule(node);
  isl_union_pw_multi_aff *contraction = isl_schedule_node_get_subtree_contraction(node);
  p_sc = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(p_sc, contraction);

  isl_bool is_uniform = isl_bool_true;
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    isl_union_pw_aff *p_sc_hyp = isl_multi_union_pw_aff_get_union_pw_aff(p_sc, i);
    /* Obtain the schedule for the src statment. */
    isl_space *space = isl_basic_map_get_space(dep);
    isl_space *src_space = isl_space_domain(isl_space_copy(space));
    isl_space *dest_space = isl_space_range(space);

    isl_pw_aff *src_sc;
    isl_pw_aff_list *p_sc_hyp_list = isl_union_pw_aff_get_pw_aff_list(p_sc_hyp);
    for (int j = 0; j < isl_union_pw_aff_n_pw_aff(p_sc_hyp); j++) {
      isl_pw_aff *single_sc = isl_pw_aff_list_get_pw_aff(p_sc_hyp_list, j);
      isl_space *single_sc_stmt = isl_space_domain(isl_pw_aff_get_space(single_sc));
      if (isl_space_is_equal(src_space, single_sc_stmt)) {
        isl_space_free(single_sc_stmt);
        src_sc = single_sc;
        break;
      }
      isl_pw_aff_free(single_sc);
      isl_space_free(single_sc_stmt);
    }
    isl_pw_aff_list_free(p_sc_hyp_list);
    isl_space_free(src_space);

    /* Obtain the schedule for the dest statement. */
    isl_pw_aff *dest_sc;
    p_sc_hyp_list = isl_union_pw_aff_get_pw_aff_list(p_sc_hyp);
    for (int j = 0; j < isl_union_pw_aff_n_pw_aff(p_sc_hyp); j++) {
      isl_pw_aff *single_sc = isl_pw_aff_list_get_pw_aff(p_sc_hyp_list, j);
      isl_space *single_sc_stmt = isl_space_domain(isl_pw_aff_get_space(single_sc));
      if (isl_space_is_equal(dest_space, single_sc_stmt)) {
        isl_space_free(single_sc_stmt);
        dest_sc = single_sc;
        break;
      }
      isl_pw_aff_free(single_sc);
      isl_space_free(single_sc_stmt);
    }
    isl_pw_aff_list_free(p_sc_hyp_list);
    isl_space_free(dest_space);

    /* Compute the dependence distance at the current hyperplane. */ 
    /* Step 1: Extend the scheduling function. */
    isl_size src_sc_dim = isl_pw_aff_dim(src_sc, isl_dim_in);
    isl_size dest_sc_dim = isl_pw_aff_dim(dest_sc, isl_dim_in);
    src_sc = isl_pw_aff_insert_dims(src_sc, isl_dim_in, src_sc_dim, dest_sc_dim);
    dest_sc = isl_pw_aff_insert_dims(dest_sc, isl_dim_in, 0, src_sc_dim);
    for (int j = 0; j < dest_sc_dim; j++) {
      isl_pw_aff_set_dim_id(src_sc, isl_dim_in, src_sc_dim + j, isl_pw_aff_get_dim_id(dest_sc, isl_dim_in, src_sc_dim + j));
    }
    for (int j = 0; j < src_sc_dim; j++) {
      isl_pw_aff_set_dim_id(dest_sc, isl_dim_in, j, isl_pw_aff_get_dim_id(src_sc, isl_dim_in, j));
    }    

    isl_pw_aff *dis_sc = isl_pw_aff_sub(dest_sc, src_sc);

    /* Step 2: Convert the basic_map into basic_set. */
    isl_mat *eq_mat = isl_basic_map_equalities_matrix(dep,
        isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
    isl_mat *ieq_mat = isl_basic_map_inequalities_matrix(dep,
        isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);

    isl_basic_set *dep_set = isl_basic_set_from_constraint_matrices(
      isl_space_domain(isl_pw_aff_get_space(dis_sc)),
      eq_mat, ieq_mat,
      isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);

    /* Step 3: Intersect the scheduling function with the domain. */
    isl_pw_aff *dis = isl_pw_aff_intersect_domain(dis_sc, 
        isl_set_from_basic_set(isl_basic_set_copy(dep_set)));

    isl_union_pw_aff_free(p_sc_hyp);
    isl_basic_set_free(dep_set);

    /* Examine if the dependence distance is constant. */
    if (!isl_pw_aff_is_cst(dis)) {
      is_uniform = isl_bool_false;
      isl_pw_aff_free(dis);
      break;
    }

    isl_pw_aff_free(dis);
  }
  
  isl_multi_union_pw_aff_free(p_sc);
  return is_uniform;
}

void print_mat(FILE *fp, __isl_keep isl_mat *mat) 
{
  isl_printer *printer = isl_printer_to_file(isl_mat_get_ctx(mat), fp);
  for (int i = 0; i < isl_mat_rows(mat); i++) {
    for (int j = 0; j < isl_mat_cols(mat); j++) {
      isl_printer_print_val(printer, isl_mat_get_element_val(mat, i, j));
      fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }
  isl_printer_free(printer);
}

/* Apply the schedule on the dependence and check if every dimension is a constant. 
 * Dep in the form of S1[]->S2[].
 */
isl_bool is_dep_uniform(__isl_take isl_basic_map *bmap, void *user)
{
  isl_bool is_uniform;
  isl_schedule *schedule = (isl_schedule *)(user);
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_ctx *ctx = isl_basic_map_get_ctx(bmap);
//  /* Search for the first permutable node and analyze the dep. */
//  is_uniform = isl_schedule_node_every_descendant(root,
//      &is_dep_uniform_at_node, bmap);

  /* Get the full schedule and apply the schedule to both the domain and range of the dependence.
   * Generate the set from this map, and apply a map that calculate the diff at each dimension to 
   * get the dependence vector. At last, check if the dependence vector is a constant vector.
   */
  isl_union_map *full_sched = isl_schedule_node_get_subtree_schedule_union_map(root);
//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_union_map(p, full_sched);
//  printf("\n");
//  p = isl_printer_print_basic_map(p, bmap);
//  printf("\n");
//  // debug
  isl_union_map *dep_tmp = isl_union_map_apply_domain(isl_union_map_from_map(isl_map_from_basic_map(bmap)), isl_union_map_copy(full_sched));
  isl_union_map *dep = isl_union_map_apply_range(dep_tmp, full_sched);

  isl_schedule_node_free(root);

  isl_map *dep_map = isl_map_from_union_map(dep);
  isl_basic_map *dep_bmap = isl_basic_map_from_map(isl_map_copy(dep_map));

  isl_set *src_dep_domain = isl_map_domain(isl_map_copy(dep_map));
  isl_map *src_dep_domain_map = isl_set_identity(src_dep_domain);
  isl_multi_pw_aff *src_mpa = isl_multi_pw_aff_identity(isl_map_get_space(src_dep_domain_map));
  isl_map_free(src_dep_domain_map);

  isl_set *dest_dep_domain = isl_map_range(dep_map);
  isl_map *dest_dep_domain_map = isl_set_identity(dest_dep_domain);
  isl_multi_pw_aff *dest_mpa = isl_multi_pw_aff_identity(isl_map_get_space(dest_dep_domain_map));
  isl_map_free(dest_dep_domain_map);

//  // debug
//  p = isl_printer_print_multi_pw_aff(p, src_mpa);
//  printf("\n");
//  p = isl_printer_print_multi_pw_aff(p, dest_mpa);
//  printf("\n");
//  // debug

  /* Add dims */
  isl_size src_dim = isl_multi_pw_aff_dim(src_mpa, isl_dim_in);
  isl_size dest_dim = isl_multi_pw_aff_dim(dest_mpa, isl_dim_in);
  src_mpa = isl_multi_pw_aff_insert_dims(src_mpa, isl_dim_in, src_dim, dest_dim);
  dest_mpa = isl_multi_pw_aff_insert_dims(dest_mpa, isl_dim_in, 0, src_dim);

//  // debug
//  p = isl_printer_print_multi_pw_aff(p, src_mpa);
//  printf("\n");
//  p = isl_printer_print_multi_pw_aff(p, dest_mpa);
//  printf("\n");
//  // debug
 
  isl_multi_pw_aff *dep_dis_mpa = isl_multi_pw_aff_sub(dest_mpa, src_mpa);
//  // debug
//  p = isl_printer_print_multi_pw_aff(p, dep_dis_mpa);
//  printf("\n");
//  // debug

  /* Convert the basic map to basic_set */
  isl_mat *eq_mat = isl_basic_map_equalities_matrix(dep_bmap,
      isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
  isl_mat *ieq_mat = isl_basic_map_inequalities_matrix(dep_bmap,
      isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
  isl_basic_set *dep_bset = isl_basic_set_from_constraint_matrices(
      isl_space_domain(isl_multi_pw_aff_get_space(dep_dis_mpa)),
      eq_mat, ieq_mat,
      isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);

  dep_dis_mpa = isl_multi_pw_aff_intersect_domain(dep_dis_mpa, isl_set_from_basic_set(dep_bset));
//  // debug
//  p = isl_printer_print_multi_pw_aff(p, dep_dis_mpa);
//  printf("\n");
//  // debug
  is_uniform = isl_multi_pw_aff_is_cst(dep_dis_mpa);

  isl_multi_pw_aff_free(dep_dis_mpa);
  isl_basic_map_free(dep_bmap);

//  isl_basic_map_free(bmap);
//  isl_schedule_node_free(root);

  return is_uniform;
}

isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user) 
{
  isl_bool is_uniform;
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  for (int i = 0; i < isl_map_n_basic_map(map); i++) {
    is_uniform = is_dep_uniform(isl_basic_map_list_get_basic_map(bmap_list, i), user);
    if (is_uniform != isl_bool_true) {
      isl_basic_map *dep_i = isl_basic_map_list_get_basic_map(bmap_list, i);
      /* Print out the non-uniform dependence. */
      isl_printer *p = isl_printer_to_file(isl_map_get_ctx(map), stdout);
      p = isl_printer_print_basic_map(p, dep_i);
      printf("\n");
      isl_printer_free(p);
      isl_basic_map_free(dep_i);

      isl_basic_map_list_free(bmap_list);
      return isl_bool_false;
    }
  }
  isl_basic_map_list_free(bmap_list);
  return isl_bool_true;
}

/* Examine if all flow and rar dependences are uniform in the program. */ 
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop) {
  isl_union_map *dep_rar = scop->dep_rar;
  isl_union_map *dep_flow = scop->dep_flow;

  isl_bool all_flow_dep_uniform = isl_union_map_every_map(dep_flow, &is_dep_uniform_wrap, schedule);
  if (all_flow_dep_uniform != isl_bool_true)
    return isl_bool_false;

  isl_bool all_rar_dep_uniform = isl_union_map_every_map(dep_rar, &is_dep_uniform_wrap, schedule);
  if (all_rar_dep_uniform != isl_bool_true)
    return isl_bool_false;  

  return isl_bool_true;
}

/* A program is legal to be transformed to systolic array if and on if 
 * it satisfies the following constraints:
 * - one single fully permutable outermost band
 * - uniform dependency
 */
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop) {
  /* Check if the root node point to a band node */
  isl_bool single_p_band;
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  node = isl_schedule_node_child(node, 0);
  enum isl_schedule_node_type type;
  type = isl_schedule_node_get_type(node);
  single_p_band = (type == isl_schedule_node_band);
  isl_schedule_node_free(node);
  if (!single_p_band) {
    printf("[PolySA] Single outermost permutable band not found.\n");
    return isl_bool_false;
  }

  /* Check if all flow and rar dependences are uniform. */
  isl_bool all_uniform_dep = uniform_dep_check(schedule, scop);
  if (all_uniform_dep < 1) {
    printf("[PolySA] Non-uniform dependence detected.\n");
    return isl_bool_false;
  }

  return isl_bool_true;
}

/* Set *depth (initialized to 0 by the caller) to the maximum
 * of the schedule depths of the leaf nodes for which this function is called.
 */
static isl_bool update_depth(__isl_keep isl_schedule_node *node, void *user)
{
	int *depth = user;
	int node_depth;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;
	node_depth = isl_schedule_node_get_schedule_depth(node);
	if (node_depth > *depth)
		*depth = node_depth;

	return isl_bool_false;
}

__isl_give isl_vec *get_dep_dis_at_schedule(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_ctx *ctx = isl_basic_map_get_ctx(dep);
  isl_union_map *full_sched = isl_schedule_node_get_subtree_schedule_union_map(root);
  isl_schedule_node_free(root);

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_union_map(p, full_sched);
//  printf("\n");
//  isl_printer_free(p);
//  // debug

  /* Extract the iterator num. */
  int iter_num = 0;
  isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth, &iter_num);

  isl_union_map *dep_sched = isl_union_map_apply_domain(isl_union_map_from_map(isl_map_from_basic_map(isl_basic_map_copy(dep))),
      isl_union_map_copy(full_sched));
  dep_sched = isl_union_map_apply_range(dep_sched, full_sched);

  isl_map *dep_map = isl_map_from_union_map(dep_sched);
  isl_basic_map *dep_bmap = isl_basic_map_from_map(isl_map_copy(dep_map));

  isl_set *src_dep_domain = isl_map_domain(isl_map_copy(dep_map));
  isl_map *src_dep_domain_map = isl_set_identity(src_dep_domain);
  isl_multi_pw_aff *src_mpa = isl_multi_pw_aff_identity(isl_map_get_space(src_dep_domain_map));
  isl_map_free(src_dep_domain_map);

  isl_set *dest_dep_domain = isl_map_range(dep_map);
  isl_map *dest_dep_domain_map = isl_set_identity(dest_dep_domain);
  isl_multi_pw_aff *dest_mpa = isl_multi_pw_aff_identity(isl_map_get_space(dest_dep_domain_map));
  isl_map_free(dest_dep_domain_map);

  /* Add dims. */
  isl_size src_dim = isl_multi_pw_aff_dim(src_mpa, isl_dim_in);
  isl_size dest_dim = isl_multi_pw_aff_dim(dest_mpa, isl_dim_in);
  src_mpa = isl_multi_pw_aff_insert_dims(src_mpa, isl_dim_in, src_dim, dest_dim);
  dest_mpa = isl_multi_pw_aff_insert_dims(dest_mpa, isl_dim_in, 0, src_dim);

  isl_multi_pw_aff *dep_dis_mpa = isl_multi_pw_aff_sub(dest_mpa, src_mpa);

  /* Convert the basic map to basic_set. */
  isl_mat *eq_mat = isl_basic_map_equalities_matrix(dep_bmap,
      isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
  isl_mat *ieq_mat = isl_basic_map_inequalities_matrix(dep_bmap,
      isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
  isl_basic_set *dep_bset = isl_basic_set_from_constraint_matrices(
      isl_space_domain(isl_multi_pw_aff_get_space(dep_dis_mpa)),
      eq_mat, ieq_mat,
      isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);

  dep_dis_mpa = isl_multi_pw_aff_intersect_domain(dep_dis_mpa, isl_set_from_basic_set(isl_basic_set_copy(dep_bset)));
//  // debug
//  p = isl_printer_print_multi_pw_aff(p, dep_dis_mpa);
//  printf("\n");
//  // debug

  isl_space *space = isl_multi_pw_aff_get_space(dep_dis_mpa);
  isl_vec *dep_dis = isl_vec_zero(ctx, isl_space_dim(space, isl_dim_out));
  for (int i = 0; i < isl_vec_size(dep_dis); i++) {
    isl_pw_aff *pa = isl_multi_pw_aff_get_pw_aff(dep_dis_mpa, i);
    isl_val *val = isl_pw_aff_eval(pa, isl_basic_set_sample_point(isl_basic_set_copy(dep_bset)));
    dep_dis = isl_vec_set_element_val(dep_dis, i, val);
  }

//  // debug
//  p = isl_printer_print_vec(p, dep_dis);
//  printf("\n");
//  // debug

  isl_space_free(space);
  isl_basic_set_free(dep_bset);
  isl_basic_map_free(dep_bmap);
  isl_multi_pw_aff_free(dep_dis_mpa);

  return dep_dis;
}

/* Compute the dependence distance vector of the dependence under the partial schedule of the band node. */
__isl_give isl_vec *get_dep_dis_at_node(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule_node *band)
{
  if (isl_schedule_node_get_type(band) != isl_schedule_node_band)
    return NULL;

  isl_multi_union_pw_aff *p_sc = isl_schedule_node_band_get_partial_schedule(band);
  isl_union_pw_multi_aff *contraction = isl_schedule_node_get_subtree_contraction(band);
  p_sc = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(p_sc, contraction);

  int band_w = isl_schedule_node_band_n_member(band);
  isl_vec *dep_dis = isl_vec_zero(isl_basic_map_get_ctx(dep), band_w);
  for (int i = 0; i < band_w; i++) {
    isl_union_pw_aff *p_sc_hyp = isl_multi_union_pw_aff_get_union_pw_aff(p_sc, i);
    /* Obtain the schedule for the src statement. */
    isl_space *space = isl_basic_map_get_space(dep);
    isl_space *src_space = isl_space_domain(isl_space_copy(space));
    isl_space *dest_space = isl_space_range(space);

    isl_pw_aff *src_sc = NULL;
    isl_pw_aff_list *p_sc_hyp_list = isl_union_pw_aff_get_pw_aff_list(p_sc_hyp);
    for (int j = 0; j < isl_union_pw_aff_n_pw_aff(p_sc_hyp); j++) {
      isl_pw_aff *single_sc = isl_pw_aff_list_get_pw_aff(p_sc_hyp_list, j);
      isl_space *single_sc_stmt = isl_space_domain(isl_pw_aff_get_space(single_sc));

      if (isl_space_is_equal(src_space, single_sc_stmt)) {
        isl_space_free(single_sc_stmt);
        src_sc = single_sc;
        break;
      }
      isl_pw_aff_free(single_sc);
      isl_space_free(single_sc_stmt);
    }
    isl_pw_aff_list_free(p_sc_hyp_list);
    isl_space_free(src_space);

    /* Obtain the schedule for the dest statement. */
    isl_pw_aff *dest_sc = NULL;
    p_sc_hyp_list = isl_union_pw_aff_get_pw_aff_list(p_sc_hyp);
    for (int j = 0; j < isl_union_pw_aff_n_pw_aff(p_sc_hyp); j++) {
      isl_pw_aff *single_sc = isl_pw_aff_list_get_pw_aff(p_sc_hyp_list, j);
      isl_space *single_sc_stmt = isl_space_domain(isl_pw_aff_get_space(single_sc));

      if (isl_space_is_equal(dest_space, single_sc_stmt)) {
        isl_space_free(single_sc_stmt);
        dest_sc = single_sc;
        break;
      }
      isl_pw_aff_free(single_sc);
      isl_space_free(single_sc_stmt);
    }
    isl_pw_aff_list_free(p_sc_hyp_list);
    isl_space_free(dest_space);

    /* Compute the dependence distance at the current hyperplane. */
    /* Step 1: Extend the scheduling function. */
    isl_size src_sc_dim = isl_pw_aff_dim(src_sc, isl_dim_in);
    isl_size dest_sc_dim = isl_pw_aff_dim(dest_sc, isl_dim_in);
    src_sc = isl_pw_aff_insert_dims(src_sc, isl_dim_in, src_sc_dim, dest_sc_dim);
    dest_sc = isl_pw_aff_insert_dims(dest_sc, isl_dim_in, 0, src_sc_dim);
    for (int j = 0; j < dest_sc_dim; j++) {
      isl_pw_aff_set_dim_id(src_sc, isl_dim_in, src_sc_dim + j, isl_pw_aff_get_dim_id(dest_sc, isl_dim_in, src_sc_dim + j));
    }
    for (int j = 0; j < src_sc_dim; j++) {
      isl_pw_aff_set_dim_id(dest_sc, isl_dim_in, j, isl_pw_aff_get_dim_id(src_sc, isl_dim_in, j));
    }    

    isl_pw_aff *dis_sc = isl_pw_aff_sub(dest_sc, src_sc);

    /* Step 2: Convert the basic_map into basic_set. */
    isl_mat *eq_mat = isl_basic_map_equalities_matrix(dep,
        isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
    isl_mat *ieq_mat = isl_basic_map_inequalities_matrix(dep,
        isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);

    isl_basic_set *dep_set = isl_basic_set_from_constraint_matrices(
      isl_space_domain(isl_pw_aff_get_space(dis_sc)),
      eq_mat, ieq_mat,
      isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);

    /* Step 3: Intersect the scheduling function with the domain. */
    isl_pw_aff *dis = isl_pw_aff_intersect_domain(dis_sc, isl_set_from_basic_set(isl_basic_set_copy(dep_set)));
    isl_val *val = isl_pw_aff_eval(dis, isl_basic_set_sample_point(dep_set));
    dep_dis = isl_vec_set_element_val(dep_dis, i, val);

    isl_union_pw_aff_free(p_sc_hyp);
  }
  
  isl_multi_union_pw_aff_free(p_sc);
  return dep_dis;
}

/* Generate asynchronized systolic arrays with the given dimension. 
 * For async arrays, space loops are placed outside the time loops.
 */
struct polysa_prog **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa) 
{
  struct polysa_prog **sas = NULL;  

  /* Select space loop candidates.
   * Space loops carry dependences with distance less or equal to 1.
   */
  isl_schedule_node *band = get_outermost_permutable_node(schedule);
  isl_size band_w = isl_schedule_node_band_n_member(band);
  isl_size *is_space_loop = (isl_size *)malloc(band_w * sizeof(isl_size));
  isl_union_map *dep_flow = scop->dep_flow;
  isl_union_map *dep_rar = scop->dep_rar;
  isl_union_map *dep_total = isl_union_map_union(isl_union_map_copy(dep_flow), isl_union_map_copy(dep_rar));

  isl_basic_map_list *deps = isl_union_map_get_basic_map_list(dep_total);
  isl_size ndeps = isl_union_map_n_basic_map(dep_total);

  for (int h = 0; h < band_w; h++) {
    int n;
    for (n = 0; n < ndeps; n++) {
      isl_basic_map *dep = isl_basic_map_list_get_basic_map(deps, n);
      isl_vec *dep_dis = get_dep_dis_at_node(dep, band);
      isl_val *val = isl_vec_get_element_val(dep_dis, h);
      if (!(isl_val_is_one(val) || isl_val_is_zero(val))) {
        isl_vec_free(dep_dis);
        isl_val_free(val);
        isl_basic_map_free(dep);
        break;         
      }

      isl_val_free(val);
      isl_vec_free(dep_dis);
      isl_basic_map_free(dep);
    }
    is_space_loop[h] = (n == ndeps);
  }

  /* Perform loop permutation to generate all candidates. */
//  // debug
//  for (int i = 0; i < band_w; i++)
//    printf("%d ", is_space_loop[i]);
//  printf("\n");
//  // debug
  if (dim == 1) {
    for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        isl_schedule *new_schedule = isl_schedule_copy(schedule);       
        /* Make the loop i the outermost loop. */
        for (int d = i; d > 0; d--) {
//          // debug
//          isl_printer *printer = isl_printer_to_file(isl_schedule_get_ctx(new_schedule), stdout);
//          isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
//          isl_printer_print_schedule(printer, new_schedule);
//          printf("\n");
//          // debug
          isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
          isl_schedule_free(new_schedule);
          new_schedule = loop_interchange_at_node(band, d, d - 1);
//          // debug
//          isl_printer_print_schedule(printer, new_schedule);
//          printf("\n");
//          // debug
        }

        /* Update the hyperplane types. */
        struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
        sa->scop = scop;
        sa->type = POLYSA_SA_TYPE_ASYNC;

        /* Update the array dimension. */
        sa->array_dim = dim;
        sa->array_part_w = 0;
        sa->space_w = dim;
        sa->time_w = band_w - dim;

        /* Add the new variant into the list. */
        sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
        sas[*num_sa] = sa;
        *num_sa = *num_sa + 1;
      } 
    }
  } else if (dim == 2) {
    for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        for (int j = i + 1; j < band_w; j++) {
          if (is_space_loop[j]) {
            isl_schedule *new_schedule = isl_schedule_copy(schedule);
            /* Make the loop i, j the outermost loops. */
            for (int d = j; d > 0; d--) {
              isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
              isl_schedule_free(new_schedule);
              new_schedule = loop_interchange_at_node(band, d, d - 1);
            }
            for (int d = i + 1; d > 0; d--) {
              isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
              isl_schedule_free(new_schedule);
              new_schedule = loop_interchange_at_node(band, d, d - 1);
            }

            /* Update the hyperplane types. */
            struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
            sa->scop = scop;
            sa->type = POLYSA_SA_TYPE_ASYNC;

            /* Update the array dimension. */
            sa->array_dim = dim;
            sa->array_part_w = 0;
            sa->space_w = dim;
            sa->time_w = band_w - dim;

            /* Add the new variant into the list. */
            sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
            sas[*num_sa] = sa;
            *num_sa = *num_sa + 1;
          }
        }
      }
    }
  } else if (dim == 3) {
     for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        for (int j = i + 1; j < band_w; j++) {
          if (is_space_loop[j]) {
            for (int k = j + 1; k < band_w; k++) {
              if (is_space_loop[k]) {
                isl_schedule *new_schedule = isl_schedule_copy(schedule);
                /* Make the loop i, j, k the outermost loops. */
                for (int d = k; d > 0; d--) {
                  isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d - 1);             
                }
                for (int d = j + 1; d > 0; d--) {
                  isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d - 1);
                }
                for (int d = i + 2; d > 0; d--) {
                  isl_schedule_node *band = get_outermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d - 1);
                }
    
                /* Update the hyperplane types. */
                struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
                sa->scop = scop;
                sa->type = POLYSA_SA_TYPE_ASYNC;

                /* Update the array dimension. */
                sa->array_dim = dim;
                sa->array_part_w = 0;
                sa->space_w = dim;
                sa->time_w = band_w - dim;
    
                /* Add the new variant into the list. */
                sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
                sas[*num_sa] = sa;
                *num_sa = *num_sa + 1;
              }
            }
          }
        }
      }
    }   
  }

  isl_basic_map_list_free(deps);
  isl_union_map_free(dep_total);
  isl_schedule_node_free(band);
  free(is_space_loop);

  return sas;
}

/* Interchange the loop at level1 and level2 in the schedule node and returns the new schedule. */
__isl_give isl_schedule *loop_interchange_at_node(__isl_take isl_schedule_node *node, isl_size level1, isl_size level2)
{
  /* Obtain the partial schedule of the node. */
  isl_multi_union_pw_aff *sc = isl_schedule_node_band_get_partial_schedule(node);
  
  /* Exchange the schedule at level1 and level2. */
  isl_multi_union_pw_aff *new_sc = isl_multi_union_pw_aff_copy(sc);
  new_sc = isl_multi_union_pw_aff_set_union_pw_aff(new_sc, level1, isl_multi_union_pw_aff_get_union_pw_aff(sc, level2));
  new_sc = isl_multi_union_pw_aff_set_union_pw_aff(new_sc, level2, isl_multi_union_pw_aff_get_union_pw_aff(sc, level1));

  /* Insert a new schedule node with the new schedule. */
  isl_bool *coincident = (isl_bool *)malloc(isl_schedule_node_band_n_member(node) * sizeof(isl_bool));
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    coincident[i] = isl_schedule_node_band_member_get_coincident(node, i);
  }
  node = isl_schedule_node_insert_partial_schedule(node, new_sc);
 
  /* Update the properties of the new node. */
  node = isl_schedule_node_band_set_permutable(node, 1);
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    node = isl_schedule_node_band_member_set_coincident(node, i, coincident[i]);
  }
  free(coincident);

  /* Delete the old node after the current node */
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_delete(node);

  /* Obtain the schedule from the schedule node. */
  isl_schedule *schedule = isl_schedule_node_get_schedule(node);

  isl_schedule_node_free(node); 
  isl_multi_union_pw_aff_free(sc);

  return schedule;
}

/* Generate syncrhonized systolic arrays with the given dimension.
 * For sync arrays, time loops are placed outside the space loops.
 */
struct polysa_prog **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa)
{
  struct polysa_prog **sas = NULL;  

  /* Select space loop candidates.
   * Space loops carry dependences with distance less or equal to 1.
   */
  isl_schedule_node *band = get_innermost_permutable_node(schedule);
  isl_size band_w = isl_schedule_node_band_n_member(band);
  isl_size *is_space_loop = (isl_size *)malloc(band_w * sizeof(isl_size));
  isl_union_map *dep_flow = scop->dep_flow;
  isl_union_map *dep_rar = scop->dep_rar;
  isl_union_map *dep_total = isl_union_map_union(isl_union_map_copy(dep_flow), isl_union_map_copy(dep_rar));

  isl_basic_map_list *deps = isl_union_map_get_basic_map_list(dep_total);
  isl_size ndeps = isl_union_map_n_basic_map(dep_total);

  for (int h = 0; h < band_w; h++) {
    int n;
    for (n = 0; n < ndeps; n++) {
      isl_basic_map *dep = isl_basic_map_list_get_basic_map(deps, n);
      isl_vec *dep_dis = get_dep_dis_at_node(dep, band);
      isl_val *val = isl_vec_get_element_val(dep_dis, h);
      if (!(isl_val_is_one(val) || isl_val_is_zero(val))) {
        isl_vec_free(dep_dis);
        isl_val_free(val);
        isl_basic_map_free(dep);
        break;         
      }

      isl_val_free(val);
      isl_vec_free(dep_dis);
      isl_basic_map_free(dep);
    }
    is_space_loop[h] = (n == ndeps);
  }

  /* Perform loop permutation to generate all candidates. */
  if (dim == 1) {
    for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        isl_schedule *new_schedule = isl_schedule_copy(schedule);       
        /* Make the loop i the innermost loop. */
        for (int d = i; d < band_w - 1; d++) {
          isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
          isl_schedule_free(new_schedule);
          new_schedule = loop_interchange_at_node(band, d, d + 1);
        }

        /* Update the hyperplane types. */
        struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
        sa->scop = scop;
        sa->type = POLYSA_SA_TYPE_SYNC;

        /* Update the array dimension. */
        sa->array_dim = dim;
        sa->array_part_w = 0;
        sa->space_w = dim;
        sa->time_w = band_w - dim;

        /* Add the new variant into the list. */
        sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
        sas[*num_sa] = sa;
        *num_sa = *num_sa + 1;
      } 
    }
  } else if (dim == 2) {
    for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        for (int j = i + 1; j < band_w; j++) {
          if (is_space_loop[j]) {
            isl_schedule *new_schedule = isl_schedule_copy(schedule);
            /* Make the loop i, j the innermost loops. */
            for (int d = i; d < band_w - 1; d++) {
              isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
              isl_schedule_free(new_schedule);
              new_schedule = loop_interchange_at_node(band, d, d + 1);
            }
            for (int d = j - 1; d < band_w - 1; d++) {
              isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
              isl_schedule_free(new_schedule);
              new_schedule = loop_interchange_at_node(band, d, d + 1);
            }

            /* Update the hyperplane types. */
            struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
            sa->scop = scop;
            sa->type = POLYSA_SA_TYPE_SYNC;

            /* Update the array dimension. */
            sa->array_dim = dim;
            sa->array_part_w = 0;
            sa->space_w = dim;
            sa->time_w = band_w - dim;

            /* Add the new variant into the list. */
            sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
            sas[*num_sa] = sa;
            *num_sa = *num_sa + 1;
          }
        }
      }
    }
  } else if (dim == 3) {
     for (int i = 0; i < band_w; i++) {
      if (is_space_loop[i]) {
        for (int j = i + 1; j < band_w; j++) {
          if (is_space_loop[j]) {
            for (int k = j + 1; k < band_w; k++) {
              if (is_space_loop[k]) {
                isl_schedule *new_schedule = isl_schedule_copy(schedule);
                /* Make the loop i, j, k the innermost loops. */
                for (int d = i; d < band_w - 1; d++) {
                  isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d + 1);                    
                }
                for (int d = j - 1; d < band_w - 1; d++) {
                  isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d + 1);
                }
                for (int d = k - 2; d < band_w - 1; d++) {
                  isl_schedule_node *band = get_innermost_permutable_node(new_schedule);
                  isl_schedule_free(new_schedule);
                  new_schedule = loop_interchange_at_node(band, d, d + 1);
                }
    
                /* Update the hyperplane types. */
                struct polysa_prog *sa = polysa_prog_from_schedule(new_schedule);
                sa->scop = scop;
                sa->type = POLYSA_SA_TYPE_SYNC;

                /* Update the array dimension. */
                sa->array_dim = dim;
                sa->array_part_w = 0;
                sa->space_w = dim;
                sa->time_w = band_w - dim;
    
                /* Add the new variant into the list. */
                sas = (struct polysa_prog **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_prog *));
                sas[*num_sa] = sa;
                *num_sa = *num_sa + 1;
              }
            }
          }
        }
      }
    }   
  }

  isl_basic_map_list_free(deps);
  isl_union_map_free(dep_total);
  isl_schedule_node_free(band);
  free(is_space_loop);

  return sas;
}

struct polysa_prog **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, 
    isl_size dim, isl_size *num_sa)
{
  if (scop->options->sa_type == POLYSA_SA_TYPE_ASYNC) {
    return sa_space_time_transform_at_dim_async(schedule, scop, dim, num_sa);
  } else if (scop->options->sa_type == POLYSA_SA_TYPE_SYNC) {
    return sa_space_time_transform_at_dim_sync(schedule, scop, dim, num_sa);
  }
}

/* Extracts the outermost permutable band node from the schedule tree.
 * When there are multiple nodes at the same level, extract the first one.
 */
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_schedule_node *t_node = NULL;
  isl_schedule_node_foreach_descendant_top_down(root,
      &is_outermost_permutable_node_update, &t_node);

  isl_schedule_node_free(root);
  return t_node;
}

/* Extract the innermost permutable band node from the schedule tree.
 * When there are multiple nodes at the same level, extract the first one.
 */
__isl_give isl_schedule_node *get_innermost_permutable_node(__isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_schedule_node *t_node = NULL;
  isl_schedule_node_foreach_descendant_top_down(root,
      &is_innermost_permutable_node_update, &t_node);

  isl_schedule_node_free(root);
  return t_node;
}

/* Apply space-time transformation to generate different systolic array candidates. */
struct polysa_prog **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size *num_sa) 
{
  struct polysa_prog **sa_list = NULL;
  isl_size n_sa = 0;

  isl_schedule_node *band = get_outermost_permutable_node(schedule);
  isl_size band_w = isl_schedule_node_band_n_member(band); 
  /* Explore 1D systolic array */
  if (scop->options->max_sa_dim >= 1 && band_w >= 1) {
    printf("[PolySA] Explore 1D systolic array.\n");
    isl_size n_sa_dim = 0;
    struct polysa_prog **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 1, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_prog **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_prog *));
    for (int i = 0; i < n_sa_dim; i++) {
      sa_list[n_sa + i] = sa_dim_list[i];
    }
    free(sa_dim_list);
    n_sa += n_sa_dim;
  }
  /* Explore 2D systolic array */
  if (scop->options->max_sa_dim >= 2 && band_w >= 2) {
    printf("[PolySA] Explore 2D systolic array.\n");
    isl_size n_sa_dim = 0;
    struct polysa_prog **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 2, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_prog **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_prog *));
    for (int i = 0; i < n_sa_dim; i++) {
      sa_list[n_sa + i] = sa_dim_list[i];
    }
    free(sa_dim_list);
    n_sa += n_sa_dim;
  }
  /* Explore 3D systolic array */
  if (scop->options->max_sa_dim >= 3 && band_w >= 3) {
    printf("[PolySA] Explore 3D systolic array.\n");
    isl_size n_sa_dim = 0;
    struct polysa_prog **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 3, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_prog **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_prog *));
    for (int i = 0; i < n_sa_dim; i++) {
      sa_list[n_sa + i] = sa_dim_list[i];
    }
    free(sa_dim_list);
    n_sa += n_sa_dim;
  }

//  // temp
//  sa_list = (struct polysa_sa **)realloc(sa_list, 1 * sizeof(struct polysa_sa *))  ;
//  sa_list[0] = isl_schedule_copy(schedule);
//  n_sa = 1;
//  // temp

  isl_schedule_free(schedule);
  isl_schedule_node_free(band);
  *num_sa = n_sa;
  return sa_list;
}

/* Select one systolic array design based on heuristics. */
struct polysa_prog *sa_candidates_smart_pick(struct polysa_prog **sa_list, __isl_keep isl_size num_sa)
{
  assert(num_sa > 0);
  struct polysa_prog *sa_opt = polysa_prog_copy(sa_list[0]);
    
  for (int i = 0; i < num_sa; i++)
    polysa_prog_free(sa_list[i]);
  free(sa_list);

  return sa_opt;
}

/* Initialize the space_time and pe_opt to polysa_loop_default for all band nodes. */
static __isl_give isl_schedule_node *init_band_node_sa_properties(__isl_take isl_schedule_node *node, void *user) 
{
  if (!node)
    return NULL;

  struct polysa_prog *sa = user;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    int band_w = isl_schedule_node_band_n_member(node);
    /* Initialize the SA properties. */
    for (int i = 0; i < band_w; i++) {
      node = isl_schedule_node_band_member_set_space_time(node, i, polysa_loop_default);
      node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
    }
  }

  return node;
}

/* Initialize the fields of time_space and pe_opt for each band node in the schedule tree. */
isl_stat sa_loop_init(struct polysa_prog *sa)
{
  isl_schedule *schedule = sa->schedule;
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  root = isl_schedule_node_map_descendant_bottom_up(root, 
      &init_band_node_sa_properties, sa);

  schedule = isl_schedule_node_get_schedule(root);
  isl_schedule_node_free(root);
  isl_schedule_free(sa->schedule);
  sa->schedule = schedule;

  return isl_stat_ok;
}

static __isl_give isl_union_map *extract_sizes_from_str(isl_ctx *ctx, const char *str)
{
  if (!str)
    return NULL;
  return isl_union_map_read_from_str(ctx, str);
}

/* Apply PE optimization including:
 * - latency hiding
 * - SIMD vectorization
 * - array partitioning
 */
isl_stat sa_pe_optimize(struct polysa_prog *sa)
{
  /* Prepartion before starting the optimization. */
  /* Initialize the polysa_loop_types. */
  sa_loop_init(sa);
  /* Extract the tile sizes. */
  sa->sizes = extract_sizes_from_str(sa->ctx, sa->scop->options->sa_sizes);
  /* Set the kernel id. */
  sa->kernel_id = 0;

  /* Array partitioning. */
  sa_array_partitioning_optimize(sa);
  /* Latency hiding. */
  sa_latency_hiding_optimize(sa);
  /* SIMD vectorization. */
  sa_SIMD_vectorization_optimize(sa);
}

static isl_schedule_node *detect_latency_hiding_loop(__isl_take isl_schedule_node *node, void *user)
{
  struct polysa_prog *sa = user;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      if (isl_schedule_node_band_member_get_coincident(node, i)) {
        node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_latency);
      }
    }
  }

  return node;
}

/* Apply latency hiding. 
 * Go through all the loops, if there is any parallel loop (considering only RAW), 
 * such a loop will be identified as latency hiding loop candidate. Such loops will be
 * tiled. The point loops will be permuted as the innermost time loops.
 */
isl_stat sa_latency_hiding_optimize(struct polysa_prog *sa)
{
  printf("[PolySA] Apply latency hiding.\n");
  isl_schedule *schedule = sa->schedule;
  /* Detect all candidate loops. */
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &detect_latency_hiding_loop, sa);
  // debug
  isl_printer *p = isl_printer_to_file(sa->ctx, stdout);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule(p, schedule);
  printf("\n");
  isl_printer_free(p);
  // debug
  /* Apply latency hiding on the candidate loops. */
  /* First extract the tile sizes from the user specification. */

  return isl_stat_ok;
}

/* Apply SIMD vectorization. 
 * Go through all the loops, if there is any vectorizable loop (parallel or reduction loop
 * with stride-0/1 access), such a loop will be identified as SIMD loop candidate. We will rank
 * the loops by heuristics and pick up one loop to be tiled. The point loops will be permuated 
 * as the innermost loops to be unrolled.
 */
isl_stat sa_SIMD_vectorization_optimize(struct polysa_prog *sa)
{
  printf("[PolySA] Apply SIMD vectorization.\n");
  isl_schedule *schedule = sa->schedule; 

  return isl_stat_ok;
}

/* Internal data structure for extract_size_of_type.
 * "type" specifies the name of the space that we want to extract.
 * "res" is used to store the subset of that space.
 */
struct polysa_extract_size_data {
	const char *type;
	isl_set *res;
};

/* This function is called for each set in a union_set.
 * If the name of the set matches data->type, we store the
 * set in data->res.
 */
static isl_stat extract_size_of_type(__isl_take isl_set *size, void *user)
{
  struct polysa_extract_size_data *data = user;
  const char *name;

  name = isl_set_get_tuple_name(size);
  if (name && !strcmp(name, data->type)) {
    data->res = size;
    return isl_stat_error;
  }

  isl_set_free(size);
  return isl_stat_ok;
}

/* Given a union map { kernel[i] -> *[...] },
 * return the range in the space called "type" for the kernel with 
 * sequence number "id".
 */
static __isl_give isl_set *extract_sa_sizes(__isl_keep isl_union_map *sizes,
    const char *type, int id)
{
  isl_space *space;
  isl_set *dom;
  isl_union_set *local_sizes;
  struct polysa_extract_size_data data = { type, NULL};

  if (!sizes)
    return NULL;

  space = isl_union_map_get_space(sizes);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, 1);
  space = isl_space_set_tuple_name(space, isl_dim_set, "kernel");
  dom = isl_set_universe(space);
  dom = isl_set_fix_si(dom, isl_dim_set, 0, id);

  local_sizes = isl_union_set_apply(isl_union_set_from_set(dom),
      isl_union_map_copy(sizes));
  isl_union_set_foreach_set(local_sizes, &extract_size_of_type, &data);
  isl_union_set_free(local_sizes);
  return data.res;
}

/* Given a singleton set, extract the *len elements of the single integer tuple
 * into *sizes. 
 *
 * If the element value is "-1", the loop at the same position is not tiled.
 *  
 * If "set" is NULL, then the "sizes" array is not updated.
 */
static isl_stat read_sa_sizes_from_set(__isl_take isl_set *set, int *sizes, int *len)
{
  int i;
  int dim;

  if (!set)
    return isl_stat_ok;

  dim = isl_set_dim(set, isl_dim_set);
  if (dim < *len)
    isl_die(isl_set_get_ctx(set), isl_error_invalid, 
        "fewer sa_sizes than required", return isl_stat_error);

  for (i = 0; i < *len; ++i) {
    isl_val *v;

    v = isl_set_plain_get_val_if_fixed(set, isl_dim_set, i);
    if (!v)
      goto error;
    sizes[i] = isl_val_get_num_si(v);
    isl_val_free(v);
  }

  isl_set_free(set);
  return isl_stat_ok;
error:
  isl_set_free(set);
  return isl_stat_error;
}

/* Add the map { kernel[id] -> type[sizes] } to gen->used-sizes 
 * if the option debug->dump_sa_sizes is set.
 */
static void set_sa_used_sizes(struct polysa_prog *sa, const char *type, int id,
    int *sizes, int len)
{
// TODO
}

/* Extract user specified "sa_tile" sizes from the "sa_sizes" command line option,
 * defaulting to option->sa_tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and 
 * return a pointer to the tile sizes (or NULL on error).
 * And the effectively used sizes to sa->used_sizes.
 */
static int *read_array_part_tile_sizes(struct polysa_prog *sa, int *tile_len)
{
  int n;
  int *tile_size;
  isl_set *size;

  tile_size = isl_alloc_array(sa->ctx, int, *tile_len);
  if (!tile_size)
    return NULL;
  for (n = 0; n < *tile_len; ++n)
    tile_size[n] = sa->scop->options->sa_tile_size;
  
  size = extract_sa_sizes(sa->sizes, "array_part", sa->kernel_id);
  if (read_sa_sizes_from_set(size, tile_size, tile_len) < 0)
    goto error;
  set_sa_used_sizes(sa, "array_part", sa->kernel_id, tile_size, *tile_len);

  return tile_size;
error:
  free(tile_size);
  return NULL;
}

static __isl_give isl_multi_val *multi_val_from_int_list(
  __isl_take isl_space *space, int *list)
{
  int i, n;
  isl_ctx *ctx;
  isl_multi_val *mv;

  if (!space) 
    return NULL;

  ctx = isl_space_get_ctx(space);
  n = isl_space_dim(space, isl_dim_set);
  mv = isl_multi_val_zero(space);
  for (i = 0; i < n; ++i) {
    isl_val *v;

    v = isl_val_int_from_si(ctx, list[i]);
    mv = isl_multi_val_set_val(mv, i, v);
  }

  return mv;
}

static __isl_give isl_multi_val *construct_band_tile_sizes(
  __isl_keep isl_schedule_node *node, int *tile_size)
{
  isl_space *space;

  if (!node)
    return NULL;

  space = isl_schedule_node_band_get_space(node);
  return multi_val_from_int_list(space, tile_size);
}

/* Tile "band" with tile size specified by "sizes".
 */
static __isl_give isl_schedule_node *tile_band(
  __isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes)
{
  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  int scale_tile;
  int shift_point;

  scale_tile = isl_options_get_tile_scale_tile_loops(ctx);
  isl_options_set_tile_scale_tile_loops(ctx, 0);
  shift_point = isl_options_get_tile_shift_point_loops(ctx);
  isl_options_set_tile_shift_point_loops(ctx, 1);

  node = isl_schedule_node_band_tile(node, sizes);

  isl_options_set_tile_scale_tile_loops(ctx, scale_tile);
  isl_options_set_tile_shift_point_loops(ctx, shift_point);
  
  return node;
}

/* Tile "band" with tile size specified by "sizes".
 *
 * If the tile size at the given position, is "-1", the loop
 * will not be tiled. Two band nodes are generated. The first band
 * contains the tile loops and the untiled loops. The second band
 * contains the point loops.
 */
__isl_give isl_schedule_node *polysa_tile_band(
  __isl_take isl_schedule_node *node, __isl_keep int *sizes)    
{
  int full_tile = 1;
  int n;

  /* Examine of the band needs to be completedly tiled. */
  n = isl_schedule_node_band_n_member(node);
  for (int i = 0; i < n; i++) {
    if (sizes[i] == -1) {
      full_tile = 0;
      break;
    }
  }

  if (full_tile) {
    isl_multi_val *tile_sizes;
    tile_sizes = construct_band_tile_sizes(node, sizes);
    node = tile_band(node, isl_multi_val_copy(tile_sizes));
    isl_multi_val_free(tile_sizes);
  } else {

  }

  return node;
}

/* Reset the pe_opt properties of all the band opts back to default. */
static __isl_give isl_schedule_node *clear_pe_opt_prop(
  __isl_take isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
    }
  }

  return node;
}

/* Apply array partitioning.
 * Apply loop tiling on the band that contains the space loops
 * Reorganize the array partitioning loops and place them following the
 * ascending order of the dependence distances. 
 */
isl_stat sa_array_partitioning_optimize(struct polysa_prog *sa)
{
  int tile_len;
  isl_schedule *schedule;
  int *tile_size;
  isl_id *id;

  printf("[PolySA] Apply array partitioning.\n");
  /* Fetch the band that contains the space loops. */
  isl_schedule_node *node;
  if (sa->type == POLYSA_SA_TYPE_SYNC) {
    node = get_innermost_permutable_node(sa->schedule);
  } else if (sa->type == POLYSA_SA_TYPE_ASYNC){
    node = get_outermost_permutable_node(sa->schedule);
  } else {
    isl_die(sa->ctx, isl_error_invalid,
    "no supported sa type", return isl_stat_error);
  }

  /* Mark the loop properties. */
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_array_part);
  }
  schedule = isl_schedule_node_get_schedule(node);

  if (sa->scop->options->debug->polysa_verbose) {
    /* Display the candidate loops. */
    isl_printer *p = isl_printer_to_file(sa->ctx, stdout);
    p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
    p = isl_printer_print_schedule(p, schedule);
    printf("\n");
    isl_printer_free(p);
  }
  isl_schedule_free(schedule);
  
  /* Tile the band. */
  tile_len = isl_schedule_node_band_n_member(node);
  tile_size = read_array_part_tile_sizes(sa, &tile_len);
  if (!tile_size) {
    isl_schedule_node_free(node);
    return isl_stat_error;
  }
  node = polysa_tile_band(node, tile_size);

  /* Add the array marker */
  node = isl_schedule_node_child(node, 0);
  id = isl_id_alloc(sa->ctx, "array", NULL);
  node = isl_schedule_node_insert_mark(node, id);
  node = isl_schedule_node_parent(node);

  // debug
  isl_printer *p_debug = isl_printer_to_file(sa->ctx, stdout);
  p_debug = isl_printer_set_yaml_style(p_debug, ISL_YAML_STYLE_BLOCK);
  p_debug = isl_printer_print_schedule_node(p_debug, node);
  printf("\n");
  isl_printer_free(p_debug);
  // debug

  /* Clean up the band pe_opt properties. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &clear_pe_opt_prop, NULL);

  isl_schedule_free(sa->schedule);
  sa->schedule = schedule;

  // debug
  p_debug = isl_printer_to_file(sa->ctx, stdout);
  p_debug = isl_printer_set_yaml_style(p_debug, ISL_YAML_STYLE_BLOCK);
  p_debug = isl_printer_print_schedule(p_debug, schedule);
  printf("\n");
  isl_printer_free(p_debug);
  // debug

  /* Clean up. */
  free(tile_size);

  return isl_stat_ok;
}

/* Free the polysa_sa struct. */
void *polysa_prog_free(struct polysa_prog *sa) 
{
  if (!sa)
    return NULL;
  isl_schedule_free(sa->schedule); 
  isl_union_map_free(sa->sizes);
  isl_union_map_free(sa->used_sizes);
  free(sa);
  return NULL;
}

/* Copy a new polysa_sa struct. */
struct polysa_prog *polysa_prog_copy(struct polysa_prog *sa) {
  struct polysa_prog *sa_dup = (struct polysa_prog *)malloc(sizeof(struct polysa_prog));
  sa_dup->ctx = sa->ctx;
  sa_dup->schedule = isl_schedule_copy(sa->schedule);
  sa_dup->scop = sa->scop;
  sa_dup->array_dim = sa->array_dim;
  sa_dup->array_part_w = sa->array_part_w;
  sa_dup->space_w = sa->space_w;
  sa_dup->time_w = sa->time_w;
  sa_dup->type = sa->type;
  sa_dup->sizes = isl_union_map_copy(sa->sizes);
  sa_dup->used_sizes = isl_union_map_copy(sa->used_sizes);
  sa_dup->kernel_id = sa->kernel_id;

  return sa_dup;
}

/* Allocate a new polysa_sa struct with the given schedule. */
struct polysa_prog *polysa_prog_from_schedule(__isl_take isl_schedule *schedule)
{
  struct polysa_prog *sa = (struct polysa_prog *)malloc(sizeof(struct polysa_prog));
  sa->ctx = isl_schedule_get_ctx(schedule);
  sa->schedule = schedule;
  sa->array_dim = 0;
  sa->array_part_w = 0;
  sa->space_w = 0;
  sa->time_w = 0;
  sa->type = 0;
  sa->sizes = NULL;
  sa->used_sizes = NULL;
  sa->kernel_id = 0;

  return sa;
}

struct polysa_prog *polysa_prog_alloc(isl_ctx *ctx, struct ppcg_scop *scop) 
{
  struct polysa_prog *prog;
  isl_space *space;
  isl_map *id;

  if (!scop)
    return NULL;

  prog = isl_calloc_type(ctx, struct polysa_prog);
  if (!prog)
    return NULL;

  prog->ctx = ctx;
  prog->scop = scop;
  prog->schedule = NULL;
  prog->sizes = NULL;
  prog->used_sizes = NULL;
}

void *polysa_acc_free(struct polysa_acc *acc) {
  if (!acc)
    return NULL;

  isl_map_free(acc->tagged_map);
  isl_map_free(acc->map);
  isl_space_free(acc->id);

  free(acc);

  return NULL;
}

__isl_null struct polysa_iter *polysa_iter_free(struct polysa_iter *iter) {
  if (!iter)
    return NULL;

  free(iter->name);
  free(iter->ts_name);
  isl_aff_free(iter->lb);
  isl_aff_free(iter->ub);

  free(iter);

  return NULL;
}
