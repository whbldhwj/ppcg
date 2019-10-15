#include "polysa_common.h"

/* Examines if the node is a permutable band node. If so, 
 * increase the number of permutable node.
 */
isl_bool is_permutable_node(__isl_keep isl_schedule_node *node, void *user) {
  isl_val *n_permutable_node = (isl_val *)(user);
  if (!node)
    return isl_bool_error;

  /* Skip the domain and leaf node. */
  if (isl_schedule_node_get_type(node) == isl_schedule_node_domain)
    return isl_bool_true;
  if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf)
    return isl_bool_true;

//  // debug
//  printf("%d\n", isl_schedule_node_get_type(node));
//  isl_printer *printer = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  isl_printer_print_schedule_node(printer, node);
//  printf("\n");
//  // debug

  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return isl_bool_false;
  if (!isl_schedule_node_band_get_permutable(node))
    return isl_bool_false;
  if (isl_schedule_node_band_n_member(node) < 1)
    return isl_bool_false;

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
      &is_permutable_node, n_permutable_node);
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
//  // debug
//  isl_printer *printer = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  isl_printer_print_multi_union_pw_aff(printer, p_sc);
//  printf("\n");
//  isl_printer_print_basic_map(printer, dep);
//  printf("\n");
//  // debug
 
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
    isl_pw_aff *dis = isl_pw_aff_intersect_domain(dis_sc, isl_set_from_basic_set(isl_basic_set_copy(dep_set)));

    // debug
//    isl_printer_print_pw_aff(printer, dis);
//    printf("\n");
//    isl_val * val = isl_pw_aff_eval(dis, isl_basic_set_sample_point(dep_set));
//    isl_printer_print_val(printer, val);
//    printf("\n");
    // debug
  
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

/* Apply the scheudle on the dependence and check if every dimension is a constant. 
 * Dep in the form of S1[]->S2[].
 */
isl_bool is_dep_uniform(__isl_take isl_basic_map *bmap, void *user)
{
  isl_schedule *schedule = (isl_schedule *)(user);
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  /* Search for the first permutable node and analyze the dep. */
  isl_stat is_uniform = isl_schedule_node_foreach_descendant_top_down(root,
      &is_dep_uniform_at_node, bmap);
  isl_schedule_node_free(root);

  isl_basic_map_free(bmap);
  return (is_uniform == isl_stat_ok ? isl_bool_true : isl_bool_false);

//  isl_schedule_node_band_get_partial_schedule_union_map
//  isl_multi_union_pw_aff_get_union_pw_aff
//  isl_union_pw_aff_add_pw_aff
//  isl_union_pw_aff_n_pw_aff
//  isl_union_pw_aff_foreach_pw_aff
//  isl_union_pw_aff_extract_pw_aff
//  isl_pw_aff_is_cst
//  isl_pw_aff_sub
//  isl_pw_aff_eval()
}

isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user) 
{
  isl_bool is_uniform;
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  for (int i = 0; i < isl_map_n_basic_map(map); i++) {
    is_uniform = is_dep_uniform(isl_basic_map_list_get_basic_map(bmap_list, i), user);
    if (is_uniform != isl_bool_true) {
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
 * - single fully permutable band
 * - uniform dependency
 */
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop) {
  // debug
  FILE *fp = fopen("schedule.tmp", "w");
  isl_printer *printer = isl_printer_to_file(isl_schedule_get_ctx(schedule), fp);
  isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  isl_printer_print_schedule(printer, schedule);
  isl_printer_free(printer);
  fclose(fp);
  // debug

  /* Check if there is only one single permutable band in the schedule tree. */
  isl_bool single_p_band = has_single_permutable_node(schedule);
  if (single_p_band < 1) {
    printf("[PSA] Single permutable band not found.\n");
    return isl_bool_false;
  }

  /* Check if all flow and rar dependences are uniform. */
  isl_bool all_uniform_dep = uniform_dep_check(schedule, scop);
  if (all_uniform_dep < 1) {
    printf("[PSA] Non-uniform dependence detected.\n");
    return isl_bool_false;
  }

  return isl_bool_true;
}
