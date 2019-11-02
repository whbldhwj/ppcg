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
 * compare and update the outermost node.
 */
isl_bool is_permutable_node_update(__isl_keep isl_schedule_node *node, void *user)
{
  isl_schedule_node **t_node = (isl_schedule_node **)(user);
  if (!node)
    return isl_bool_error;

  if (is_permutable_node(node) == isl_bool_true) {
    if (*t_node == NULL)
      *t_node = isl_schedule_node_copy(node);
    else {
      if (isl_schedule_node_get_tree_depth(node) < isl_schedule_node_get_tree_depth(*t_node)) {
        isl_schedule_node_free(*t_node);
        *t_node = isl_schedule_node_copy(node);
      } else if (isl_schedule_node_get_tree_depth(node) == isl_schedule_node_get_tree_depth(*t_node)) {
        if (isl_schedule_node_get_child_position(node) < isl_schedule_node_get_child_position(*t_node)) {
          isl_schedule_node_free(*t_node);
          *t_node = isl_schedule_node_copy(node);
        }
      }
    }
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

/* Apply the scheudle on the dependence and check if every dimension is a constant. 
 * Dep in the form of S1[]->S2[].
 */
isl_bool is_dep_uniform(__isl_take isl_basic_map *bmap, void *user)
{
  isl_schedule *schedule = (isl_schedule *)(user);
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  /* Search for the first permutable node and analyze the dep. */
  isl_bool is_uniform = isl_schedule_node_every_descendant(root,
      &is_dep_uniform_at_node, bmap);
  isl_schedule_node_free(root);

  isl_basic_map_free(bmap);
  return is_uniform;

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
//  // debug
//  FILE *fp = fopen("schedule.tmp", "w");
//  isl_printer *printer = isl_printer_to_file(isl_schedule_get_ctx(schedule), fp);
//  isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
//  isl_printer_print_schedule(printer, schedule);
//  isl_printer_free(printer);
//  fclose(fp);
//  // debug

  /* Check if there is only one single permutable band in the schedule tree. */
  isl_bool single_p_band = has_single_permutable_node(schedule);
  if (single_p_band < 1) {
    printf("[PolySA] Single permutable band not found.\n");
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

        /* Update the array dimension. */
        sa->array_dim = dim;
        sa->array_part_w = 0;

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

            /* Update the array dimension. */
            sa->array_dim = dim;
            sa->array_part_w = 0;

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

                /* Update the array dimension. */
                sa->array_dim = dim;
                sa->array_part_w = 0;
    
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
  return NULL;
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
  isl_schedule_node_every_descendant(root,
      &is_permutable_node_update, &t_node);

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

/* Apply PE optimization including:
 * - latency hiding
 * - SIMD vectorization
 * - array partitioning
 */
isl_stat sa_pe_optimize(struct polysa_prog *sa)
{
  sa_latency_hiding_optimize(sa);
  sa_SIMD_vectorization_optimize(sa);
  sa_array_partitioning_optimize(sa);
}

/* Apply latency hiding. */
isl_stat sa_latency_hiding_optimize(struct polysa_prog *sa)
{
  printf("[PolySA] Apply latency hiding.\n");

  return isl_stat_ok;
}

/* Apply SIMD vectorization. */
isl_stat sa_SIMD_vectorization_optimize(struct polysa_prog *sa)
{
  printf("[PolySA] Apply SIMD vectorization.\n");

  return isl_stat_ok;
}

/* Apply array partitioning. 
 * Apply loop tiling on the outermost permutable loops (ignore point loops for 
 * latency hiding and SIMD vectorization). 
 * Reorganize the array partitioning loops and place them following the
 * ascending order of the dependence distances. 
 */
isl_stat sa_array_partitioning_optimize(struct polysa_prog *sa)
{
//  int tile_len;
//  int tile_size;
//
  printf("[PolySA] Apply array partitioning.\n");
//  isl_schedule_node *band = get_outermost_permutable_node(sa->schedule);
//
//  tile_size = read_tile_sizes(, &tile_len);
//  if (!tile_size) {
//    isl_schedule_node_free(band);
//    return isl_stat_ok;
//  }
// 
//  isl_schedule_node_band_tile(node, sizes);
//  tile_ban
//
//  isl_schedule_free(sa->schedule);
//  sa->schedule = loop_tiling_at_node(band, );

  return isl_stat_ok;
}

/* Tile "band" with tile size specified by "sizes".
 * The tile loop scaling is turned off, and the point loop 
 * shifting is turned on.
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

/* Free the polysa_sa struct. */
void *polysa_prog_free(struct polysa_prog *sa) 
{
  if (!sa)
    return NULL;
  isl_schedule_free(sa->schedule);  
  free(sa);
  return NULL;
}

/* Copy a new polysa_sa struct. */
struct polysa_prog *polysa_prog_copy(struct polysa_prog *sa) {
  struct polysa_prog *sa_dup = (struct polysa_prog *)malloc(sizeof(struct polysa_prog));

  sa_dup->schedule = isl_schedule_copy(sa->schedule);
  sa_dup->scop = sa->scop;
  sa_dup->array_dim = sa->array_dim;
  sa_dup->array_part_w = sa->array_part_w;
  sa_dup->space_w = sa->space_w;
  sa_dup->time_w = sa->time_w;
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


