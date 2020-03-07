#include "polysa_common.h"

/********************************************************************
 * Band node related functions
 ********************************************************************/
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

__isl_give isl_multi_val *construct_band_tile_sizes(
  __isl_keep isl_schedule_node *node, int *tile_size)
{
  isl_space *space;

  if (!node)
    return NULL;

  space = isl_schedule_node_band_get_space(node);
  return multi_val_from_int_list(space, tile_size);
}

struct polysa_node_band_prop *extract_node_band_prop(__isl_keep isl_schedule_node *node)
{
  struct polysa_node_band_prop *prop = isl_calloc_type(isl_schedule_node_get_ctx(node), 
      struct polysa_node_band_prop);
  prop->mupa = isl_schedule_node_band_get_partial_schedule(node);
  prop->n_member = isl_schedule_node_band_n_member(node);
  prop->coincident = isl_calloc_array(isl_schedule_node_get_ctx(node), int, prop->n_member);
  for (int i = 0; i < prop->n_member; i++) {
    prop->coincident[i] = isl_schedule_node_band_member_get_coincident(node, i);
  }
  prop->permutable = isl_schedule_node_band_get_permutable(node);
  prop->space_time = isl_calloc_array(isl_schedule_node_get_ctx(node), 
      enum polysa_loop_type, prop->n_member);
  prop->pe_opt = isl_calloc_array(isl_schedule_node_get_ctx(node),
      enum polysa_loop_type, prop->n_member);
  for (int i = 0; i < prop->n_member; i++) {
    prop->space_time[i] = isl_schedule_node_band_member_get_space_time(node, i);
    prop->pe_opt[i] = isl_schedule_node_band_member_get_pe_opt(node, i);
  }

  return prop;
}

struct polysa_node_band_prop *polysa_node_band_prop_free(__isl_take struct polysa_node_band_prop *prop)
{
  isl_multi_union_pw_aff_free(prop->mupa);
  free(prop->coincident);
  free(prop->space_time);
  free(prop->pe_opt);

  free(prop);

  return NULL;
}

/* Examines if the node is a permutable band node. */
isl_bool is_permutable_node(__isl_keep isl_schedule_node *node) 
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
static isl_bool is_outermost_permutable_node_update(__isl_keep isl_schedule_node *node, void *user)
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

isl_bool no_permutable_node(isl_schedule_node *node, void *user)
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
static isl_bool is_innermost_permutable_node_update(__isl_keep isl_schedule_node *node, void *user)
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
static isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user) {
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

/* Compute the dependence distance vector of the dependence under the partial schedule of the band node. 
 * The dependence "dep" is untagged.
 */
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
  struct polysa_node_band_prop *prop = extract_node_band_prop(node);

//  isl_bool *coincident = (isl_bool *)malloc(isl_schedule_node_band_n_member(node) * sizeof(isl_bool));
//  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
//    coincident[i] = isl_schedule_node_band_member_get_coincident(node, i);
//  }
  node = isl_schedule_node_insert_partial_schedule(node, new_sc);
 
  /* Update the properties of the new node. */
  node = isl_schedule_node_band_set_permutable(node, 1);
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    node = isl_schedule_node_band_member_set_coincident(node, i, prop->coincident[i]);
  }
  node = isl_schedule_node_band_member_set_coincident(node, level1, prop->coincident[level2]);
  node = isl_schedule_node_band_member_set_coincident(node, level2, prop->coincident[level1]);
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    node = isl_schedule_node_band_member_set_pe_opt(node, i, prop->pe_opt[i]);
  }
  node = isl_schedule_node_band_member_set_pe_opt(node, level1, prop->pe_opt[level2]);
  node = isl_schedule_node_band_member_set_pe_opt(node, level2, prop->pe_opt[level1]);
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    node = isl_schedule_node_band_member_set_space_time(node, i, prop->space_time[i]);
  }
  node = isl_schedule_node_band_member_set_space_time(node, level1, prop->space_time[level2]);
  node = isl_schedule_node_band_member_set_space_time(node, level2, prop->space_time[level1]);

  polysa_node_band_prop_free(prop); 

  /* Delete the old node after the current node */
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_delete(node);

  /* Obtain the schedule from the schedule node. */
  isl_schedule *schedule = isl_schedule_node_get_schedule(node);

  isl_schedule_node_free(node); 
  isl_multi_union_pw_aff_free(sc);

  return schedule;
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

/* Tile "band" with tile size specified by "sizes".
 */
__isl_give isl_schedule_node *tile_band(
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
    /* Reset the space_time in the tile band */
    for (int i = 0; i < n; i++) {
      node = isl_schedule_node_band_member_set_space_time(node, i, polysa_loop_time);
    }
    isl_multi_val_free(tile_sizes);
  } else {
    // TODO: tile on demand
  }

  return node;
}

/* Reset the pe_opt properties of all the band opts back to default. */
__isl_give isl_schedule_node *clear_pe_opt_prop(
  __isl_take isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
    }
  }

  return node;
}

/* Except partial schedule, restore the rest band node properties. */
__isl_give isl_schedule_node *restore_node_band_prop(__isl_take isl_schedule_node *node, 
  __isl_take struct polysa_node_band_prop *prop) 
{
  node = isl_schedule_node_band_set_permutable(node, prop->permutable);
  for (int i = 0; i < prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_coincident(node, i, prop->coincident[i]);
  }
  for (int i = 0; i < prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_space_time(node, i, prop->space_time[i]);
    node = isl_schedule_node_band_member_set_pe_opt(node, i, prop->pe_opt[i]);
  }

  free(prop->coincident);
  free(prop->pe_opt);
  free(prop->space_time);
  isl_multi_union_pw_aff_free(prop->mupa);
  free(prop);

  return node;
}

/* Given two nested nodes,
 * N1
 * |
 * N2
 * Interchange the two nodes to
 * N2
 * |
 * N1
 * return a pointer to node N2.
 */
__isl_give isl_schedule_node *polysa_node_interchange(__isl_take isl_schedule_node *node)
{
  if (isl_schedule_node_n_children(node) == 0 || isl_schedule_node_n_children(node) > 1) {
    return node;
  }

  /* Save the current node. */
  struct polysa_node_band_prop *prop = extract_node_band_prop(node);

  /* Delete the current node. */
  node = isl_schedule_node_delete(node);

  /* Insert the old node. */
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_partial_schedule(node, isl_multi_union_pw_aff_copy(prop->mupa));

  /* Restore the node properties. */
  node = restore_node_band_prop(node, prop);
  
  node = isl_schedule_node_parent(node);

  return node;
}

/* Examines if the current schedule node is a io mark at the level "io_level".
 * Specifically, the io mark at the level "io_level" has the name as "io_L[io_level]".
 */
isl_bool isl_schedule_node_is_io_mark(__isl_keep isl_schedule_node *node, int io_level) {
  isl_id *mark;
  const char *name;
  isl_printer *p;
  char *io_mark;

  if (!node)
    return isl_bool_error;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_mark)
    return isl_bool_false;

  mark = isl_schedule_node_mark_get_id(node);
  if (!mark)
    return isl_bool_error;

  name = isl_id_get_name(mark);
  p = isl_printer_to_str(isl_schedule_node_get_ctx(node));
  p = isl_printer_print_str(p, "io_L");
  p = isl_printer_print_int(p, io_level);
  io_mark = isl_printer_get_str(p);
  p = isl_printer_free(p);
  isl_id_free(mark);
  if (!strcmp(name, io_mark)) {
    free(io_mark);
    return isl_bool_true;
  } else {
    free(io_mark);
    return isl_bool_false;
  }
}

/***************************************************************
 * PolySA kernel related functions 
 ***************************************************************/
/* Free the polysa_sa struct. */
void *polysa_kernel_free(struct polysa_kernel *kernel) 
{
  if (!kernel)
    return NULL;
  
  isl_schedule_free(kernel->schedule);
  isl_ast_node_free(kernel->tree);
  isl_union_map_free(kernel->sizes);
  isl_union_map_free(kernel->used_sizes);
  isl_union_set_free(kernel->core);
  isl_set_free(kernel->context);
  isl_multi_pw_aff_free(kernel->sa_grid_size);
  isl_union_set_free(kernel->arrays);
  isl_union_pw_multi_aff_free(kernel->copy_schedule);
  isl_space_free(kernel->space);
  isl_id_list_free(kernel->block_ids);
  isl_id_list_free(kernel->thread_ids);
  isl_id_list_free(kernel->pe_ids);
  isl_union_set_free(kernel->pe_filter);
  isl_multi_pw_aff_free(kernel->grid_size);
  isl_ast_expr_free(kernel->grid_size_expr);
  isl_union_pw_multi_aff_free(kernel->contraction);
  isl_union_set_free(kernel->expanded_domain);
  isl_set_free(kernel->host_domain);
  isl_union_set_free(kernel->domain);
  for (int i = 0; i < kernel->n_array; ++i) {
    struct polysa_local_array_info *array = &kernel->array[i];
    for (int j = 0; j < array->n_pe_group; ++j)
      polysa_array_ref_group_free(array->pe_groups[j]);
    free(array->pe_groups);
    for (int j = 0; j < array->n_io_group; ++j)
      polysa_array_ref_group_free(array->io_groups[j]);
    free(array->io_groups);
    polysa_array_ref_group_free(array->drain_group);

    isl_multi_pw_aff_free(array->bound);
    isl_ast_expr_free(array->bound_expr);
  }
  if (kernel->array)
    free(kernel->array);

  for (int i = 0; i < kernel->n_var; i++) {
    free(kernel->var[i].name);
    isl_vec_free(kernel->var[i].size);
  }
  free(kernel->var);

  free(kernel);
  return NULL;
}

/* Copy a new polysa_sa struct. */
struct polysa_kernel *polysa_kernel_copy(struct polysa_kernel *sa) 
{
  struct polysa_kernel *sa_dup = (struct polysa_kernel *)malloc(sizeof(struct polysa_kernel));
  sa_dup->ctx = sa->ctx;
  sa_dup->schedule = isl_schedule_copy(sa->schedule);
  sa_dup->scop = sa->scop;
  sa_dup->options = sa->options;
  sa_dup->n_sa_dim = sa->n_sa_dim;
  for (int i = 0; i < sa->n_sa_dim; i++) {
    sa_dup->sa_dim[i] = sa->sa_dim[i];
  }
  sa_dup->array_part_w = sa->array_part_w;
  sa_dup->space_w = sa->space_w;
  sa_dup->time_w = sa->time_w;
  sa_dup->type = sa->type;
  sa_dup->sa_grid_size = isl_multi_pw_aff_copy(sa->sa_grid_size);
  sa_dup->sizes = isl_union_map_copy(sa->sizes);
  sa_dup->used_sizes = isl_union_map_copy(sa->used_sizes);
  sa_dup->id = sa->id;
  sa_dup->core = isl_union_set_copy(sa->core);
  sa_dup->arrays = isl_union_set_copy(sa->arrays);
  sa_dup->n_array = sa->n_array;
  sa_dup->array = sa->array;
  sa_dup->copy_schedule = isl_union_pw_multi_aff_copy(sa->copy_schedule);
  sa_dup->copy_schedule_dim = sa->copy_schedule_dim;
  sa_dup->space = isl_space_copy(sa->space);
  sa_dup->tree = isl_ast_node_copy(sa->tree);
  sa_dup->n_var = sa->n_var;
  sa_dup->var = sa->var;
  sa_dup->block_ids = isl_id_list_copy(sa->block_ids);
  sa_dup->thread_ids = isl_id_list_copy(sa->thread_ids);
  sa_dup->pe_ids = isl_id_list_copy(sa->pe_ids);
  sa_dup->pe_filter = isl_union_set_copy(sa->pe_filter);
  sa_dup->n_grid = sa->n_grid;
  sa_dup->n_block = sa->n_block;
  for (int i = 0; i < sa->n_grid; i++) {
    sa_dup->grid_dim[i] = sa->grid_dim[i];
  }
  for (int i = 0; i < sa->n_block; i++) {
    sa_dup->block_dim[i] = sa->block_dim[i];
  }
  sa_dup->grid_size = isl_multi_pw_aff_copy(sa->grid_size);
  sa_dup->grid_size_expr = isl_ast_expr_copy(sa->grid_size_expr);
  sa_dup->context = isl_set_copy(sa->context);
  sa_dup->contraction = isl_union_pw_multi_aff_copy(sa->contraction);
  sa_dup->expanded_domain = isl_union_set_copy(sa->expanded_domain);
  sa_dup->host_domain = isl_set_copy(sa->host_domain);
  sa_dup->domain = isl_union_set_copy(sa->domain);
  sa_dup->single_statement = sa->single_statement;

  return sa_dup;
}

/* Allocate a new polysa_sa struct with the given schedule. */
struct polysa_kernel *polysa_kernel_from_schedule(__isl_take isl_schedule *schedule)
{
  struct polysa_kernel *kernel = (struct polysa_kernel *)malloc(sizeof(struct polysa_kernel));
  kernel->ctx = isl_schedule_get_ctx(schedule);
  kernel->schedule = schedule;
  kernel->scop = NULL;
  kernel->prog = NULL;
  kernel->options = NULL;
  kernel->n_sa_dim = 0;
  kernel->array_part_w = 0;
  kernel->space_w = 0;
  kernel->time_w = 0;
  kernel->type = 0;
  kernel->sa_grid_size = NULL;
  kernel->sizes = NULL;
  kernel->used_sizes = NULL;
  kernel->id = 0;
  kernel->core = NULL;
  kernel->arrays = NULL;
  kernel->n_array = 0;
  kernel->array = NULL;
  kernel->copy_schedule = NULL;
  kernel->copy_schedule_dim = -1;
  kernel->space = NULL;
  kernel->tree = NULL;
  kernel->n_var = 0;
  kernel->var = NULL;
  kernel->block_ids = NULL;
  kernel->thread_ids = NULL;
  kernel->pe_ids = NULL;
  kernel->pe_filter = NULL;
  kernel->n_grid = 0;
  kernel->n_block = 0;
  kernel->grid_size = NULL;
  kernel->grid_size_expr = NULL;
  kernel->context = NULL;
  kernel->contraction = NULL;
  kernel->expanded_domain = NULL;
  kernel->host_domain = NULL;
  kernel->domain = NULL;
  kernel->single_statement = 0;

  return kernel;
}

struct polysa_kernel *polysa_kernel_alloc(isl_ctx *ctx, struct ppcg_scop *scop) 
{
  struct polysa_kernel *kernel;
  isl_space *space;
  isl_map *id;

  if (!scop)
    return NULL;

  kernel = isl_calloc_type(ctx, struct polysa_kernel);
  if (!kernel)
    return NULL;

  kernel->ctx = ctx;
  kernel->scop = scop;
  kernel->prog = NULL;
  kernel->options = NULL;
  kernel->n_sa_dim = 0;
  kernel->array_part_w = 0;
  kernel->space_w = 0;
  kernel->time_w = 0;
  kernel->type = 0;
  kernel->sa_grid_size = NULL;
  kernel->sizes = NULL;
  kernel->used_sizes = NULL;
  kernel->id = 0;
  kernel->core = NULL;
  kernel->arrays = NULL;
  kernel->n_array = 0;
  kernel->array = NULL;
  kernel->copy_schedule = NULL;
  kernel->copy_schedule_dim = -1;
  kernel->space = NULL;
  kernel->tree = NULL;
  kernel->n_var = 0;
  kernel->var = NULL;
  kernel->block_ids = NULL;
  kernel->thread_ids = NULL;
  kernel->pe_ids = NULL;
  kernel->pe_filter = NULL;
  kernel->n_grid = 0;
  kernel->n_block = 0;
  kernel->grid_size = NULL;
  kernel->grid_size_expr = NULL;
  kernel->context = NULL;
  kernel->contraction = NULL;
  kernel->expanded_domain = NULL;
  kernel->host_domain = NULL;
  kernel->domain = NULL;
  kernel->single_statement = 0;

  return kernel;
}

/********************************************************************
 * Other PolySA structs
 ********************************************************************/
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

/********************************************************************
 * PolySA dep related functions
 ********************************************************************/
/* Free up the dependence. */
void *polysa_dep_free(__isl_take struct polysa_dep *dep)
{
  if (!dep)
    return NULL;

  if (dep->src)
    dep->src = isl_id_free(dep->src);
  if (dep->dest)
    dep->dest = isl_id_free(dep->dest);
  if (dep->disvec)
    isl_vec_free(dep->disvec);
  if (dep->src_sched_domain)
    isl_set_free(dep->src_sched_domain);
  if (dep->dest_sched_domain)
    isl_set_free(dep->dest_sched_domain);
  if (dep->isl_dep)
    isl_basic_map_free(dep->isl_dep);

  free(dep);

  return NULL;
}

/**********************************************************************
 * Schedule related functions
 **********************************************************************/
/* Construct schedule constraints from the dependences in prog->scop and
 * the array order dependences in prog->array_order.
 *
 * If live range reordering is allowed, then we need to make sure
 * that live ranges on arrays are not run in parallel since doing
 * so would require array expansion.  We therefore add the array
 * order dependences to the coincidence dependences.  Non-zero array
 * order dependences will then prevent a schedule dimension from being
 * considered parallel.
 * Live ranges derived from scalars are allowed to be run in parallel
 * since we force the scalars to be mapped to private memory in
 * check_scalar_live_ranges.
 * If live range reordering is allowed, then the false dependences
 * are not added to the validity constraints as that would prevent
 * reordering.  Instead, the external false dependences that enforce that reads
 * from potentially live-in data precede any later write and
 * that writes of potentially live-out data follow any other earlier write
 * are added to the validity and the coincidence constraints.
 * The false dependences are still added to the proximity constraints
 * for consistency with the case where live range reordering is not allowed.
 * The coincidence constraints then consist of flow dependences,
 * external false dependences and array order dependences.
 * The independences can be filtered out from the first two sets.
 * They have already been filtered out from the array order dependences
 * on a per array basis in collect_order_dependences.
 * There is no need for a per array handling of the other two sets
 * as there should be no flow or external false dependence on local
 * variables that can be filtered out.
 */
static __isl_give isl_schedule_constraints *construct_schedule_constraints(
	struct polysa_prog *prog)
{
	isl_union_set *domain;
	isl_union_map *dep_raw, *dep;
	isl_union_map *validity, *proximity, *coincidence;
	isl_schedule_constraints *sc;

	domain = isl_union_set_copy(prog->scop->domain);
	sc = isl_schedule_constraints_on_domain(domain);
	sc = isl_schedule_constraints_set_context(sc,
				isl_set_copy(prog->scop->context));
	if (prog->scop->options->live_range_reordering) {
		sc = isl_schedule_constraints_set_conditional_validity(sc,
			isl_union_map_copy(prog->scop->tagged_dep_flow),
			isl_union_map_copy(prog->scop->tagged_dep_order));
		proximity = isl_union_map_copy(prog->scop->dep_flow);
		validity = isl_union_map_copy(proximity);
		validity = isl_union_map_union(validity,
			    isl_union_map_copy(prog->scop->dep_forced));
		proximity = isl_union_map_union(proximity,
			    isl_union_map_copy(prog->scop->dep_false));
		coincidence = isl_union_map_copy(validity);
		coincidence = isl_union_map_subtract(coincidence,
			isl_union_map_copy(prog->scop->independence));
		coincidence = isl_union_map_union(coincidence,
				isl_union_map_copy(prog->array_order));
    /* Add the RAR into the validity constraints for PolySA. */
    if (prog->scop->options->polysa) {
      validity = isl_union_map_union(validity,
          isl_union_map_copy(prog->scop->dep_rar));
    }
	} else {
		dep_raw = isl_union_map_copy(prog->scop->dep_flow);
		dep = isl_union_map_copy(prog->scop->dep_false);
		dep = isl_union_map_union(dep, dep_raw);
		dep = isl_union_map_coalesce(dep);
		proximity = isl_union_map_copy(dep);
		coincidence = isl_union_map_copy(dep);
		validity = dep;
    /* Add the RAR into the validity constraints for PolySA. */
    if (prog->scop->options->polysa) {
      validity = isl_union_map_union(validity,
          isl_union_map_copy(prog->scop->dep_rar));
    }   
	}
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);

	return sc;
}

/* Compute an appropriate schedule based on the accesses in
 * gen->read and gen->write.
 *
 * We derive schedule constraints from the dependences in gen->prog->scop
 * and then use isl to compute a schedule that has a parallel loop
 * in each tilable band.
 * During the schedule construction, some statement instances
 * may be grouped first based on the input schedule.
 */
__isl_give isl_schedule *compute_schedule(struct polysa_gen *gen)
{
	isl_schedule_constraints *sc;
	isl_schedule *schedule;

	sc = construct_schedule_constraints(gen->prog);
	schedule = gen->prog->scop->schedule;
	schedule = ppcg_compute_schedule(sc, schedule, gen->options);

	return schedule;
}

/* If the band node "node" has exactly one member then mark it permutable.
 */
static __isl_give isl_schedule_node *band_set_permutable(
	__isl_take isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	if (isl_schedule_node_band_n_member(node) == 1)
		node = isl_schedule_node_band_set_permutable(node, 1);

	return node;
}

/* Return the coincidence constraints between pairs of instances
 * that are scheduled together by the ancestors of "node".
 * That is, select those coincidence constraints that relate
 * pairs of instances that have the same value for the prefix schedule.
 * If the schedule depth is zero, then the prefix schedule does not
 * contain any information, so we intersect domain and range
 * of the schedule constraints with the reaching domain elements instead.
 */
static __isl_give isl_union_map *get_local_coincidence(
	__isl_keep isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_union_map *coincidence;
	isl_multi_union_pw_aff *prefix;
	isl_union_pw_multi_aff *contraction;

	coincidence = isl_schedule_constraints_get_coincidence(sc);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	if (isl_schedule_node_get_schedule_depth(node) == 0) {
		isl_union_set *domain;

		domain = isl_schedule_node_get_domain(node);
		domain = isl_union_set_preimage_union_pw_multi_aff(domain,
						    contraction);
		coincidence = isl_union_map_intersect_domain(coincidence,
						    isl_union_set_copy(domain));
		coincidence = isl_union_map_intersect_range(coincidence,
						    domain);
		return coincidence;
	}

	prefix = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
	prefix = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(prefix,
								contraction);
	return isl_union_map_eq_at_multi_union_pw_aff(coincidence, prefix);
}

/* For each member in the band node "node", determine whether
 * it is coincident with respect to the outer nodes and mark
 * it accordingly.
 *
 * That is, for each coincidence constraint between pairs
 * of instances that are scheduled together by the outer nodes,
 * check that domain and range are assigned the same value
 * by the band member.  This test is performed by checking
 * that imposing the same value for the band member does not
 * remove any elements from the set of coincidence constraints.
 */
static __isl_give isl_schedule_node *band_set_coincident(
	__isl_take isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_union_map *coincidence;
	isl_union_pw_multi_aff *contraction;
	isl_multi_union_pw_aff *partial;
	int i, n;

	coincidence = get_local_coincidence(node, sc);

	partial = isl_schedule_node_band_get_partial_schedule(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
								contraction);
	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i) {
		isl_union_map *coincidence_i;
		isl_union_pw_aff *upa;
		isl_multi_union_pw_aff *partial_i;
		int subset;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(partial, i);
		partial_i = isl_multi_union_pw_aff_from_union_pw_aff(upa);
		coincidence_i = isl_union_map_copy(coincidence);
		coincidence_i = isl_union_map_eq_at_multi_union_pw_aff(
						    coincidence_i, partial_i);
		subset = isl_union_map_is_subset(coincidence, coincidence_i);
		isl_union_map_free(coincidence_i);

		if (subset < 0)
			break;
		node = isl_schedule_node_band_member_set_coincident(node, i,
								    subset);
	}
	if (i < n)
		node = isl_schedule_node_free(node);
	isl_multi_union_pw_aff_free(partial);
	isl_union_map_free(coincidence);

	return node;
}

/* If "node" is a band, then set its properties.
 *
 * In particular, if the band has exactly one member, then mark it permutable.
 * Mark the band members coincident based on the coincidence constraints
 * of "sc".
 */
static __isl_give isl_schedule_node *set_band_properties(
	__isl_take isl_schedule_node *node, void *user)
{
	isl_schedule_constraints *sc = user;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;
	if (isl_schedule_node_band_n_member(node) == 0)
		return node;

	node = band_set_permutable(node, sc);
	node = band_set_coincident(node, sc);

	return node;
}

/* Return the original schedule with all bands marked permutable and
 * all band members marked coincident based on the coincidence constraints.
 * The bands are explicitly marked permutable so that they will be considered
 * by mark_outer_permutable.
 */
static __isl_give isl_schedule *determine_properties_original_schedule(
	struct polysa_gen *gen)
{
	isl_schedule *schedule;
	isl_schedule_constraints *sc;

	schedule = isl_schedule_copy(gen->prog->scop->schedule);
	sc = construct_schedule_constraints(gen->prog);
	schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
						    &set_band_properties, sc);
	isl_schedule_constraints_free(sc);

	return schedule;
}

/* Compute a schedule or determine the properties of the original schedule
 * depending on the value of the "reschedule" option.
 */
static __isl_give isl_schedule *compute_or_set_properties(void *user)
{
  struct polysa_gen *gen = user;

  if (gen->options->reschedule)
    return compute_schedule(gen);
  else
    return determine_properties_original_schedule(gen);  
}

/* Obtain a schedule for the scop, by reading it from
 * a file, by computing one or by determining the properties
 * of the original schedule. 
 */
__isl_give isl_schedule *get_schedule(struct polysa_gen *gen)
{
  return ppcg_get_schedule(gen->ctx, gen->options,
        &compute_or_set_properties, gen);
}

/****************************************************************
 * PolySA array related functions
 ****************************************************************/
/* If group->n_ref == 1, then group->refs was set by
 * populate_array_references to point directly into
 * group->array->refs and should not be freed.
 * If group->n_ref > 1, then group->refs was set by join_groups
 * to point to a newly allocated array.
 */
struct polysa_array_ref_group *polysa_array_ref_group_free(
	struct polysa_array_ref_group *group)
{
	if (!group)
		return NULL;
  polysa_array_tile_free(group->local_tile);
	isl_map_free(group->access);
	if (group->n_ref > 1)
		free(group->refs);
  isl_vec_free(group->dir);
  isl_multi_aff_free(group->io_trans);
  isl_multi_aff_free(group->io_L1_trans);
//  isl_mat_free(group->io_trans_mat);
  isl_ast_expr_free(group->io_pe_expr);
  isl_ast_expr_free(group->io_L1_pe_expr);
  for (int i = 0; i < group->n_io_buffer; i++) {
    polysa_array_tile_free(group->io_buffers[i]->tile);
    free(group->io_buffers[i]);
  }
  free(group->io_buffers);
  isl_schedule_free(group->io_schedule);
  isl_schedule_free(group->io_L1_schedule);
	free(group);

	return NULL;
}

static void *free_polysa_io_info(struct polysa_io_info *io_info) 
{
  polysa_dep_free(io_info->dep);
  isl_vec_free(io_info->dir);
  isl_vec_free(io_info->old_dir);

  free(io_info);
  return NULL;
}

static void free_array_info(struct polysa_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		free(prog->array[i].type);
		free(prog->array[i].name);
		isl_multi_pw_aff_free(prog->array[i].bound);
		isl_ast_expr_free(prog->array[i].bound_expr);
		isl_space_free(prog->array[i].space);
		isl_set_free(prog->array[i].declared_extent);
		isl_set_free(prog->array[i].extent);
		isl_ast_expr_free(prog->array[i].declared_size);
		free(prog->array[i].refs);
		isl_union_map_free(prog->array[i].dep_order);
	}
	free(prog->array);
}

/* Does "kernel" need to be passed an argument corresponding to array "i"?
 *
 * The argument is only needed if the kernel accesses this device memory.
 */
int polysa_kernel_requires_array_argument(struct polysa_kernel *kernel, int i)
{
	return kernel->array[i].global;
}

/* Is the array "array" being extracted a read-only scalar?
 *
 * That is, is "array" a scalar that is never possibly written to.
 * An array containing structures is never considered to be a scalar.
 */
static int is_read_only_scalar(struct polysa_array_info *array,
	struct polysa_prog *prog)
{
	isl_set *space;
	isl_union_map *write;
	int empty;

	if (array->has_compound_element)
		return 0;
	if (array->n_index != 0)
		return 0;

	write = isl_union_map_copy(prog->may_write);
	space = isl_set_universe(isl_space_copy(array->space));
	write = isl_union_map_intersect_range(write,
						isl_union_set_from_set(space));
	empty = isl_union_map_is_empty(write);
	isl_union_map_free(write);

	return empty;
}

/* Compute and return the extent of "array", taking into account the set of
 * accessed elements.
 *
 * In particular, the extent in the outer dimension is taken
 * from "accessed", while the extents in the remaining dimensions
 * are taken from array->extent.
 *
 * The extent in the outer dimension cannot be taken from array->extent
 * because that may be unbounded.  Furthermore, even if it is bounded,
 * it may be larger than the piece of the array that is being accessed.
 */
static __isl_give isl_set *compute_extent(struct pet_array *array,
	__isl_keep isl_set *accessed)
{
	int n_index;
	isl_id *id;
	isl_set *outer;
	isl_set *extent;

	extent = isl_set_copy(array->extent);

	n_index = isl_set_dim(accessed, isl_dim_set);
	if (n_index == 0)
		return extent;

	extent = isl_set_project_out(extent, isl_dim_set, 0, 1);
	outer = isl_set_copy(accessed);
	outer = isl_set_project_out(outer, isl_dim_set, 1, n_index - 1);
	extent = isl_set_flat_product(outer, extent);
	id = isl_set_get_tuple_id(accessed);
	extent = isl_set_set_tuple_id(extent, id);

	return extent;
}

/* Return the name of the outer array (of structs) accessed by "access".
 */
static const char *get_outer_array_name(__isl_keep isl_map *access)
{
	isl_space *space;
	const char *name;

	space = isl_space_range(isl_map_get_space(access));
	while (space && isl_space_is_wrapping(space))
		space = isl_space_domain(isl_space_unwrap(space));
	name = isl_space_get_tuple_name(space, isl_dim_set);
	isl_space_free(space);

	return name;
}

/* Collect all references to the given array and store pointers to them
 * in array->refs.
 */
static isl_stat collect_references(struct polysa_prog *prog,
	struct polysa_array_info *array)
{
	int i;
	int n;

	n = 0;
	for (i = 0; i < prog->n_stmts; ++i) {
		struct polysa_stmt *stmt = &prog->stmts[i];
		struct polysa_stmt_access *access;

		for (access = stmt->accesses; access; access = access->next) {
			const char *name;
			name = get_outer_array_name(access->access);
			if (name && !strcmp(array->name, name))
				n++;
		}
	}

	array->refs = isl_alloc_array(prog->ctx, struct polysa_stmt_access *, n);
	if (!array->refs)
		return isl_stat_error;
	array->n_ref = n;

	n = 0;
	for (i = 0; i < prog->n_stmts; ++i) {
		struct polysa_stmt *stmt = &prog->stmts[i];
		struct polysa_stmt_access *access;

		for (access = stmt->accesses; access; access = access->next) {
			const char *name;
			name = get_outer_array_name(access->access);
			if (!name || strcmp(array->name, name))
				continue;

			array->refs[n++] = access;
		}
	}

	return isl_stat_ok;
}

/* Is "array" only accessed as individual, fixed elements?
 * That is, does each access to "array" access a single, fixed element?
 */
static isl_bool only_fixed_element_accessed(struct polysa_array_info *array)
{
	int i;

	for (i = 0; i < array->n_ref; ++i)
		if (!array->refs[i]->fixed_element)
			return isl_bool_false;

	return isl_bool_true;
}

static isl_stat extract_array_info(struct polysa_prog *prog,
  struct polysa_array_info *info, struct pet_array *pa,
  __isl_keep isl_union_set *arrays)
{
  int empty;
  const char *name;
  int n_index;
  isl_multi_pw_aff *bounds;
  isl_set *accessed, *extent;

  n_index = isl_set_dim(pa->extent, isl_dim_set);
  name = isl_set_get_tuple_name(pa->extent);

  info->space = isl_set_get_space(pa->extent);
  info->name = strdup(name);
  info->n_index = n_index;
  info->linearize = prog->scop->options->linearize_device_arrays;

  info->type = strdup(pa->element_type);
  info->size = pa->element_size;
  info->local = pa->declared && !pa->exposed;
  info->has_compound_element = pa->element_is_record;
  info->read_only_scalar = is_read_only_scalar(info, prog); 

  info->declared_extent = isl_set_copy(pa->extent);
  accessed = isl_union_set_extract_set(arrays,
                isl_space_copy(info->space));
  empty = isl_set_is_empty(accessed); 
  extent = compute_extent(pa, accessed); // TODO: to understand
  isl_set_free(accessed);
  info->extent = extent;
  if (empty < 0)
    return isl_stat_error;
  info->accessed = !empty;
  bounds = ppcg_size_from_extent(isl_set_copy(extent)); 
	bounds = isl_multi_pw_aff_gist(bounds, isl_set_copy(prog->context));
	if (!bounds)
		return isl_stat_error;
	if (!isl_multi_pw_aff_is_cst(bounds))
		info->linearize = 1;
	info->bound = bounds;

	if (collect_references(prog, info) < 0) 
		return isl_stat_error;
	info->only_fixed_element = only_fixed_element_accessed(info); 

  /* PolySA Extended */
  
  /* PolySA Extended */

	return isl_stat_ok;  
}

/* Can "array" be mapped to private memory?
 * That is, is it only accessed as individual elements with
 * constant index expressions?
 */
static isl_bool polysa_array_can_be_private(struct polysa_array_info *array)
{
	if (!array)
		return isl_bool_error;
	return array->only_fixed_element;
}

/* Remove independence from the order constraints "order" on array "array".
 * Since the pairs of iterations in the filter relation of an independence
 * are guaranteed to be completely independent by the user, there is
 * no need to ensure that live ranges are ordered along those pairs.
 * We make an exception for local variables, though, as the independence
 * guarantee does not apply to those.
 *
 * The order constraints are used in two places.
 * Those on scalars are used in check_scalar_live_ranges to check if
 * we need to force the scalar to be private.  Any non-local scalar
 * should not be forced scalar if it only appears in independent loops.
 * Those on non-scalars are added to the coincidence constraints
 * in compute_schedule because we do not support any array expansion.
 * Accesses to non-local arrays should not prevent a loop from being
 * considered coincident so we should indeed remove those constraints
 * from the order constraints.
 */
static __isl_give isl_union_map *remove_independences(struct polysa_prog *prog,
	struct polysa_array_info *array, __isl_take isl_union_map *order)
{
	int i;

	for (i = 0; i < prog->scop->pet->n_independence; ++i) {
		struct pet_independence *pi = prog->scop->pet->independences[i];
		if (isl_union_set_contains(pi->local, array->space))
			continue;

		order = isl_union_map_subtract(order,
						isl_union_map_copy(pi->filter));
	}

	return order;
}

/* For each array in "prog", store the (untagged) order dependences
 * derived from the array in array->dep_order.
 * In particular, consider all references that access the given array
 * and take the order dependences that have one of these references
 * as source.  (Since an order dependence relates two references to
 * the same array, the target of these order dependences will also
 * be one of these references.)
 * Additionally, store the union of these array->dep_order relations
 * for all arrays that cannot be mapped to private memory in prog->array_order.
 */
static void collect_order_dependences(struct polysa_prog *prog)
{
	int i;
	isl_space *space;
	isl_union_map *accesses;

	space = isl_union_map_get_space(prog->read);
	prog->array_order = isl_union_map_empty(space);

	accesses = isl_union_map_copy(prog->scop->tagged_reads);
	accesses = isl_union_map_union(accesses,
			    isl_union_map_copy(prog->scop->tagged_may_writes));
	accesses = isl_union_map_universe(accesses);
	accesses = isl_union_map_apply_range(accesses,
					    isl_union_map_copy(prog->to_outer));

	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];
		isl_set *set;
		isl_union_set *uset;
		isl_union_map *order;

		set = isl_set_universe(isl_space_copy(array->space));
		uset = isl_union_set_from_set(set);
		uset = isl_union_map_domain(
		    isl_union_map_intersect_range(isl_union_map_copy(accesses),
						    uset));
		order = isl_union_map_copy(prog->scop->tagged_dep_order);
		order = isl_union_map_intersect_domain(order, uset);
		order = isl_union_map_zip(order);
		order = isl_union_set_unwrap(isl_union_map_domain(order));
		order = remove_independences(prog, array, order);
		array->dep_order = order;

		if (polysa_array_can_be_private(array)) // TODO: handle in the future
			continue;

		prog->array_order = isl_union_map_union(prog->array_order,
					isl_union_map_copy(array->dep_order));
	}

	isl_union_map_free(accesses);
}

/* Construct a polysa_array_info for each array referenced by prog->scop and
 * collect them in prog->array.
 * 
 * The sizes are based on the extents and the set of possibly accessed
 * elements by "prog".
 * If there are any member accesses involves, then they are first mapped
 * to the outer arrays of structs.
 * Only extract polysa_array_info entries for these outer arrays.
 * 
 * If we are allowing live range reordering, then also set 
 * the dep_order field. Otherwise leve it NULL.
 */
isl_stat collect_array_info(struct polysa_prog *prog)
{
  int i;
  isl_stat r = isl_stat_ok;
  isl_union_set *arrays;

  prog->n_array = 0;
  prog->array = isl_calloc_array(prog->ctx, 
          struct polysa_array_info, prog->scop->pet->n_array);
  if (!prog->array)
    return isl_stat_error;
  
  arrays = isl_union_map_range(isl_union_map_copy(prog->read));
  arrays = isl_union_set_union(arrays, 
        isl_union_map_range(isl_union_map_copy(prog->may_write)));

  arrays = isl_union_set_apply(arrays,
          isl_union_map_copy(prog->to_outer));

  arrays = isl_union_set_coalesce(arrays);

  for (i = 0; i < prog->scop->pet->n_array; ++i) {
    isl_bool field;

    field = isl_set_is_wrapping(prog->scop->pet->arrays[i]->extent);
    if (field < 0)
      break;
    if (field)
      continue;
    if (extract_array_info(prog, &prog->array[prog->n_array++],
          prog->scop->pet->arrays[i], arrays) < 0)
      r = isl_stat_error;      
  }
  if (i < prog->scop->pet->n_array)
    r = isl_stat_error;
  
  isl_union_set_free(arrays);

  if (prog->scop->options->live_range_reordering)
    collect_order_dependences(prog);
  
  return r;
}

/* Is "array" a read-only scalar?
 */
int polysa_array_is_read_only_scalar(struct polysa_array_info *array)
{
	return array->read_only_scalar;
}

/* Check if a gpu array is a scalar.  A scalar is a value that is not stored
 * as an array or through a pointer reference, but as a single data element.
 * At the moment, scalars are represented as zero-dimensional arrays.
 * Note that the single data element may be an entire structure.
 */
int polysa_array_is_scalar(struct polysa_array_info *array)
{
	return array->n_index == 0;
}

/* Compute the set of inner array elements that may have their values
 * preserved by "prog".  In particular, collect the array elements of
 * arrays that are not local to "prog" and remove those elements that
 * are definitely killed or definitely written by "prog".
 */
static __isl_give isl_union_set *compute_may_persist(struct polysa_prog *prog)
{
	int i;
	isl_union_set *may_persist, *killed;
	isl_union_map *must_kill;

	may_persist = isl_union_set_empty(isl_set_get_space(prog->context));
	for (i = 0; i < prog->n_array; ++i) {
		isl_set *extent;

		if (prog->array[i].local)
			continue;

		extent = isl_set_copy(prog->array[i].extent);
		may_persist = isl_union_set_add_set(may_persist, extent);
	}

	may_persist = isl_union_set_intersect_params(may_persist,
						isl_set_copy(prog->context));
	may_persist = isl_union_set_apply(may_persist,
					isl_union_map_copy(prog->to_inner));
	must_kill = isl_union_map_copy(prog->tagged_must_kill);
	killed = isl_union_map_range(must_kill);
	must_kill = isl_union_map_copy(prog->must_write);
	killed = isl_union_set_union(killed, isl_union_map_range(must_kill));

	may_persist = isl_union_set_subtract(may_persist, killed);
	return may_persist;
}

/*****************************************************************
 * PolySA stmts related functions
 *****************************************************************/
static void *free_stmts(struct polysa_stmt *stmts, int n)
{
  int i;

  if (!stmts)
    return NULL;

  for (i = 0; i < n; ++i) {
    struct polysa_stmt_access *access, *next;

    for (access = stmts[i].accesses; access; access = next) {
      next = access->next;
      isl_id_free(access->ref_id);
      isl_map_free(access->access);
      isl_map_free(access->tagged_access);

      for (int k = 0; k < access->n_io_info; k++)
        free_polysa_io_info(access->io_info[k]);
      free(access->io_info);

      free(access);
    }

    isl_id_free(stmts[i].id);
  }
  free(stmts);

  return NULL;
}

/* Has statement "stmt" been killed from "scop"?
 * That is, is the instance set of "scop" free from any
 * instances of "stmt"?
 */
static isl_bool is_stmt_killed(struct ppcg_scop *scop, struct pet_stmt *stmt)
{
	isl_space *space;
	isl_set *left;
	isl_bool empty;

	if (!scop || !stmt)
		return isl_bool_error;
	space = isl_set_get_space(stmt->domain);
	left = isl_union_set_extract_set(scop->domain, space);
	empty = isl_set_plain_is_empty(left);
	isl_set_free(left);

	return empty;
}

/* Given a tagged access relation to a single array "tagged", extract it
 * as a map, taking into account that the input may be empty.
 * If the access relation is empty, then it does not contain
 * any space information, so we try to recover it from the index
 * expression.
 * The space of the index expression is of the form I -> A,
 * with I the statement instances and A the array, or [I -> F] -> A,
 * with F the filters corresponding to arguments.
 * We first drop F, if present, obtaining I -> A.
 * Then we construct I -> R, with R the reference tag,
 * combine the two into I -> [R -> A] and uncurry to obtain
 * the final result [I -> R] -> A.
 * Note that the index expression may have a lower dimension
 * than that of the array, but this dimension is not used
 * if the access relation is empty.
 */
static __isl_give isl_map *extract_single_tagged_access(
	__isl_take isl_union_map *tagged, __isl_keep pet_expr *expr)
{
	int empty;
	isl_id *id;
	isl_space *space, *space2;
	isl_multi_pw_aff *index;

	empty = isl_union_map_is_empty(tagged);
	if (empty < 0)
		goto error;
	if (!empty)
		return isl_map_from_union_map(tagged);
	isl_union_map_free(tagged);

	index = pet_expr_access_get_index(expr);
	space = isl_multi_pw_aff_get_space(index);
	isl_multi_pw_aff_free(index);
	if (isl_space_domain_is_wrapping(space))
		space = isl_space_domain_factor_domain(space);
	space2 = isl_space_copy(space);
	space2 = isl_space_from_domain(isl_space_domain(space));
	id = pet_expr_access_get_ref_id(expr);
	space2 = isl_space_set_tuple_id(space2, isl_dim_out, id);
	space = isl_space_range_product(space2, space);
	space = isl_space_uncurry(space);

	return isl_map_empty(space);
error:
	isl_union_map_free(tagged);
	return NULL;
}

/* Does the index expression "index" of "expr" represent an access
 * to a single element?
 * That is, is "index" completely specified?
 *
 * If "expr" accesses elements from different spaces (i.e., fields
 * of a structure), then it does not access a single element.
 * Otherwise, if the single space of the access matches the space
 * of "index", then the index expression is completely specified
 * (no pointer to a lower-dimensional slice of the accessed array)
 * and a single element is being accessed.
 */
static isl_bool complete_index(__isl_keep pet_expr *expr,
	__isl_keep isl_multi_pw_aff *index)
{
	isl_union_map *read, *write, *all;
	isl_map *map;
	isl_space *space1, *space2;
	isl_bool complete;

	read = pet_expr_access_get_may_read(expr);
	write = pet_expr_access_get_may_write(expr);
	all = isl_union_map_union(read, write);
	if (!all)
		return isl_bool_error;
	if (isl_union_map_n_map(all) != 1) {
		isl_union_map_free(all);
		return isl_bool_false;
	}
	map = isl_map_from_union_map(all);
	space1 = isl_map_get_space(map);
	isl_map_free(map);
	space2 = isl_multi_pw_aff_get_space(index);
	complete = isl_space_tuple_is_equal(space1, isl_dim_out,
					    space2, isl_dim_out);
	isl_space_free(space1);
	isl_space_free(space2);

	return complete;
}

/* Does "expr" access a single, fixed element (independently of the statement
 * instance)?
 * That is, does it have a completely specified constant index expression?
 *
 * Note that it is not sufficient for the index expression to be
 * piecewise constant.  isl_multi_pw_aff_is_cst can therefore not be used.
 */
static isl_bool accesses_fixed_element(__isl_keep pet_expr *expr)
{
	int i, n;
	isl_multi_pw_aff *index;
	isl_bool fixed = isl_bool_true;

	index = pet_expr_access_get_index(expr);
	if (index < 0)
		return isl_bool_error;
	n = isl_multi_pw_aff_dim(index, isl_dim_out);
	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_multi_pw_aff_get_pw_aff(index, 0);
		fixed = isl_pw_aff_n_piece(pa) == 1;
		if (fixed)
			fixed = isl_pw_aff_is_cst(pa);
		isl_pw_aff_free(pa);
		if (fixed < 0 || !fixed)
			break;
	}
	if (fixed >= 0 && fixed)
		fixed = complete_index(expr, index);
	isl_multi_pw_aff_free(index);

	return fixed;
}

/* Extract a polysa_stmt_access from "expr", append it to the list
 * that ends in *data->next_access and update the end of the list.
 * If the access expression performs a write, then it is considered
 * exact only if it appears in a single expression statement and
 * if its may access relation is equal to its must access relation.
 *
 * The combined set of may accesses may be a union if member accesses
 * are involved, but the entire set is derived from a single reference and
 * therefore from a single index expression.  These accesses therefore
 * all map to the same outer array.
 */
static int extract_access(__isl_keep pet_expr *expr, void *user)
{
	struct ppcg_extract_access_data *data = user;
	isl_union_map *tagged;
	struct polysa_stmt_access *access;
	isl_ctx *ctx = pet_expr_get_ctx(expr);
	isl_multi_pw_aff *index;

	access = isl_alloc_type(ctx, struct polysa_stmt_access);
	if (!access)
		return -1;
	access->next = NULL;
	access->read = pet_expr_access_is_read(expr);
	access->write = pet_expr_access_is_write(expr);
	tagged = pet_expr_access_get_tagged_may_read(expr);
	tagged = isl_union_map_union(tagged,
				pet_expr_access_get_tagged_may_write(expr));
	tagged = isl_union_map_apply_range(tagged,
					isl_union_map_copy(data->any_to_outer));
	if (!access->write) {
		access->exact_write = 1;
	} else if (!data->single_expression) {
		access->exact_write = 0;
	} else {
		isl_union_map *must, *may;
		may = isl_union_map_copy(tagged);
		may = isl_union_map_domain_factor_domain(may);
		must = pet_expr_access_get_must_write(expr);
		access->exact_write = isl_union_map_is_equal(must, may);
		isl_union_map_free(must);
		isl_union_map_free(may);
	}
	index = pet_expr_access_get_index(expr);
	access->n_index = isl_multi_pw_aff_dim(index, isl_dim_out);
	isl_multi_pw_aff_free(index);
	access->ref_id = pet_expr_access_get_ref_id(expr);
	access->tagged_access = extract_single_tagged_access(tagged, expr);
	access->access = isl_map_copy(access->tagged_access);
	access->access = isl_map_domain_factor_domain(access->access);
	access->fixed_element = accesses_fixed_element(expr);

  access->n_io_info = 0;
  access->io_info = NULL;

	*data->next_access = access;
	data->next_access = &(*data->next_access)->next;

	if (!access->access || access->fixed_element < 0)
		return -1;

	return 0;
}

/* Construct a linked list of polysa_stmt_access objects,
 * one for each access expression in the statement body.
 * "any_to_outer" maps all intermediate arrays to their outer arrays.
 */
static int pet_stmt_extract_accesses(struct polysa_stmt *stmt,
  __isl_keep isl_union_map *any_to_outer)
{
  struct ppcg_extract_access_data data;

  stmt->accesses = NULL;
  data.next_access = &stmt->accesses;
  data.single_expression = 
    pet_tree_get_type(stmt->stmt->body) == pet_tree_expr;
  data.any_to_outer = any_to_outer;
  return pet_tree_foreach_access_expr(stmt->stmt->body,
              &extract_access, &data);
}

void polysa_kernel_stmt_free(void *user)
{
  struct polysa_kernel_stmt *stmt = user;

  if (!stmt)
    return;

  switch (stmt->type) {
    case POLYSA_KERNEL_STMT_COPY:
      isl_ast_expr_free(stmt->u.c.index);
      isl_ast_expr_free(stmt->u.c.local_index);
      break;
    case POLYSA_KERNEL_STMT_DOMAIN:
      isl_id_to_ast_expr_free(stmt->u.d.ref2expr);
      break;
    case POLYSA_KERNEL_STMT_SYNC:
      break;
    case POLYSA_KERNEL_STMT_IO:
    case POLYSA_KERNEL_STMT_IO_TRANSFER:
    case POLYSA_KERNEL_STMT_IO_TRANSFER_BUF:
    case POLYSA_KERNEL_STMT_IO_DRAM:
      free(stmt->u.i.fifo_name);
      isl_ast_expr_free(stmt->u.i.local_index);
      isl_ast_expr_free(stmt->u.i.index);
      break;
    case POLYSA_KERNEL_STMT_MODULE_CALL:
    case POLYSA_KERNEL_STMT_FIFO_DECL:
      break;
  }

  free(stmt);
}

/* Return an array of polysa_stmt representing the statements in "scop".
 * Do not collect array accesses for statements that have been killed.
 */
struct polysa_stmt *extract_stmts(isl_ctx *ctx, struct ppcg_scop *scop,
  __isl_keep isl_union_map *any_to_outer) 
{
  int i;
  struct polysa_stmt *stmts;

  stmts = isl_calloc_array(ctx, struct polysa_stmt, scop->pet->n_stmt);
  if (!stmts)
    return NULL;

  for (i = 0; i < scop->pet->n_stmt; ++i) {
    struct polysa_stmt *s = &stmts[i];
    isl_bool killed;

    s->id = isl_set_get_tuple_id(scop->pet->stmts[i]->domain);
    s->stmt = scop->pet->stmts[i];
    killed = is_stmt_killed(scop, scop->pet->stmts[i]); 
    if (killed < 0)
      return free_stmts(stmts, i + 1); 
    if (killed)
      continue;
    if (pet_stmt_extract_accesses(s, any_to_outer) < 0) 
      return free_stmts(stmts, i + 1);
  }

  return stmts;
}

/*****************************************************************
 * PolySA prog related functions
 *****************************************************************/
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
  prog->context = isl_set_copy(scop->context);
  prog->n_stmts = scop->pet->n_stmt;
  prog->any_to_outer = pet_scop_compute_outer_to_any(scop->pet);
  prog->any_to_outer = isl_union_map_reverse(prog->any_to_outer);
  space = isl_union_map_get_space(prog->any_to_outer);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, 1);
  space = isl_space_map_from_set(space);
  id = isl_map_identity(space);
  prog->any_to_outer = isl_union_map_add_map(prog->any_to_outer, id);
	prog->stmts = extract_stmts(ctx, scop, prog->any_to_outer); 
	prog->read = isl_union_map_copy(scop->reads);
	prog->may_write = isl_union_map_copy(scop->may_writes);
	prog->must_write = isl_union_map_copy(scop->must_writes);
	prog->tagged_must_kill = isl_union_map_copy(scop->tagged_must_kills);
	prog->to_inner = pet_scop_compute_outer_to_inner(scop->pet);
	prog->to_outer = isl_union_map_copy(prog->to_inner);
	prog->to_outer = isl_union_map_reverse(prog->to_outer);

  if (!prog->stmts)
    return polysa_prog_free(prog);

  if (collect_array_info(prog) < 0) 
    return polysa_prog_free(prog);
  prog->may_persist = compute_may_persist(prog); // TODO

  return prog;
}

void *polysa_prog_free(struct polysa_prog *prog)
{
	if (!prog)
		return NULL;
	free_array_info(prog);
	free_stmts(prog->stmts, prog->n_stmts);
	isl_union_map_free(prog->any_to_outer);
	isl_union_map_free(prog->to_outer);
	isl_union_map_free(prog->to_inner);
	isl_union_map_free(prog->read);
	isl_union_map_free(prog->may_write);
	isl_union_map_free(prog->must_write);
	isl_union_map_free(prog->tagged_must_kill);
	isl_union_map_free(prog->array_order);
	isl_union_set_free(prog->may_persist);
	isl_set_free(prog->context);
	free(prog);

	return NULL;
}

/*****************************************************************
 * PolySA hw module related functions
 *****************************************************************/
struct polysa_hw_module *polysa_hw_module_alloc()
{
  struct polysa_hw_module *module = (struct polysa_hw_module *)malloc(sizeof(struct polysa_hw_module));
  module->name = NULL;
  module->tree = NULL;
  module->device_tree = NULL;
  module->inst_ids = NULL;
  module->n_var = 0;
  module->var = NULL;
  module->kernel = NULL;
  module->n_io_group = 0;
  module->io_groups = NULL;
  module->to_pe = 0;
  module->to_mem = 0;

  return module;
}

struct polysa_hw_top_module *polysa_hw_top_module_alloc()
{
  struct polysa_hw_top_module *module = (struct polysa_hw_top_module *)malloc(sizeof(struct polysa_hw_top_module));

  module->n_module_calls = 0;
  module->n_fifo_decls = 0;
  module->module_call_scheds = NULL;
  module->fifo_decl_scheds = NULL;
  module->module_call_trees = NULL;
  module->fifo_decl_trees = NULL;

  module->n_module_call_wrapped = 0;
  module->n_fifo_decl_wrapped = 0;
  module->module_call_wrapped_trees = NULL;
  module->fifo_decl_wrapped_trees = NULL;

  module->kernel = NULL;
  module->hw_modules = NULL;
  module->n_hw_modules = 0;
  
  return module;
}

void *polysa_hw_module_free(struct polysa_hw_module *module)
{
  if (!module) 
    return NULL;

  free(module->name);
//  // TODO: valgrind
//  isl_schedule_free(module->sched);

  isl_ast_node_free(module->tree);
  isl_ast_node_free(module->device_tree);
  isl_id_list_free(module->inst_ids);
  for (int i = 0; i < module->n_var; i++) {
    free(module->var[i].name);
    isl_vec_free(module->var[i].size);
  }
  free(module->var);
  free(module->io_groups);
  free(module);

  return NULL;
}

void *polysa_hw_top_module_free(struct polysa_hw_top_module *module)
{
  if (!module)
    return NULL;

  if (module->module_call_trees) {
    for (int i = 0; i < module->n_module_calls; i++) {
      isl_ast_node_free(module->module_call_trees[i]);
    }
  }

  if (module->fifo_decl_trees) {
    for (int i = 0; i < module->n_fifo_decls; i++) {
      isl_ast_node_free(module->fifo_decl_trees[i]);
    }
  }

  if (module->module_call_wrapped_trees) {
    for (int i = 0; i < module->n_module_call_wrapped; i++) {
      isl_ast_node_free(module->module_call_wrapped_trees[i]);
    }
  }

  if (module->fifo_decl_wrapped_trees) {
    for (int i = 0; i < module->n_fifo_decl_wrapped; i++) {
      isl_ast_node_free(module->fifo_decl_wrapped_trees[i]);
    }
  }

//  // TODO: valgrind
//  for (int i = 0; i < module->n_hw_modules; i++) {
//    isl_schedule_free(module->scheds[i]);
//  }
  free(module->module_call_scheds);
  free(module->fifo_decl_scheds);
  free(module->module_call_trees);
  free(module->fifo_decl_trees);
  free(module->module_call_wrapped_trees);
  free(module->fifo_decl_wrapped_trees);
  free(module);

  return NULL;
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
static void set_sa_used_sizes(struct polysa_kernel *sa, const char *type, int id,
    int *sizes, int len)
{
// TODO
}

/* Extract user specified "sa_tile" sizes from the "sa_sizes" command line options,
 * defaulting to option->sa_tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and
 * return a pointer to the tile sizes (or NULL on error).
 * And the effectively used sizes to sa->used_sizes.
 */
int *read_hbm_tile_sizes(struct polysa_kernel *sa, int *tile_len)
{
  int n;
  int *tile_size;
  isl_set *size;

  tile_size = isl_alloc_array(sa->ctx, int, *tile_len);
  if (!tile_size)
    return NULL;
  for (n = 0; n < *tile_len; ++n) 
    tile_size[n] = sa->scop->options->n_hbm_port;

  size = extract_sa_sizes(sa->sizes, "hbm", sa->id);
  if (read_sa_sizes_from_set(size, tile_size, tile_len) < 0)
    goto error;
  set_sa_used_sizes(sa, "hbm", sa->id, tile_size, *tile_len);

  return tile_size;
error:
  free(tile_size);
  return NULL;
}

/* Given a union map { kernel[i] -> *[...] },
 * return the range in the space called "type" for the kernel with 
 * sequence number "id".
 */
__isl_give isl_set *extract_sa_sizes(__isl_keep isl_union_map *sizes,
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

/* Extract user specified "sa_tile" sizes from the "sa_sizes" command line option,
 * defaulting to option->sa_tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and 
 * return a pointer to the tile sizes (or NULL on error).
 * And the effectively used sizes to sa->used_sizes.
 */
int *read_array_part_tile_sizes(struct polysa_kernel *sa, int *tile_len)
{
  int n;
  int *tile_size;
  isl_set *size;

  tile_size = isl_alloc_array(sa->ctx, int, *tile_len);
  if (!tile_size)
    return NULL;
  for (n = 0; n < *tile_len; ++n)
    tile_size[n] = sa->scop->options->sa_tile_size;
  
  size = extract_sa_sizes(sa->sizes, "array_part", sa->id);
  if (read_sa_sizes_from_set(size, tile_size, tile_len) < 0)
    goto error;
  set_sa_used_sizes(sa, "array_part", sa->id, tile_size, *tile_len);

  return tile_size;
error:
  free(tile_size);
  return NULL;
}

/* Extract user specified "sa_tile" sizes from the "sa_sizes" command line option,
 * defaulting to option->sa_tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and
 * return a pointer to the tile sizes (or NULL on error).
 * And store the effectively used sizes to sa->used_sizes.
 */
int *read_latency_tile_sizes(struct polysa_kernel *sa, int *tile_len)
{
  int n;
  int *tile_size;
  isl_set *size;

  tile_size = isl_alloc_array(sa->ctx, int, *tile_len);
  if (!tile_size)
    return NULL;
  for (n = 0; n < *tile_len; n++)
    tile_size[n] = sa->scop->options->sa_tile_size / 2;

  size = extract_sa_sizes(sa->sizes, "latency", sa->id);
  if (read_sa_sizes_from_set(size, tile_size, tile_len) < 0)
    goto error;
  set_sa_used_sizes(sa, "latency", sa->id, tile_size, *tile_len);

  return tile_size;
error:
  free(tile_size);
  return NULL;
}

