#include "polysa_trans.h"
#include "polysa_device.h"
#include "polysa_array_tile.h"

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
static void set_sa_used_sizes(struct polysa_kernel *sa, const char *type, int id,
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
static int *read_array_part_tile_sizes(struct polysa_kernel *sa, int *tile_len)
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

/* Apply array partitioning.
 * Apply loop tiling on the band that contains the space loops
 * Reorganize the array partitioning loops and place them following the
 * ascending order of the dependence distances. 
 */
isl_stat sa_array_partitioning_optimize(struct polysa_kernel *sa)
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

//  // debug
//  isl_printer *p = isl_printer_to_file(sa->ctx, stdout);
//  isl_multi_union_pw_aff *mupa = isl_schedule_node_band_get_partial_schedule(node);
//  isl_space *space = isl_multi_union_pw_aff_get_space(mupa);
//  p = isl_printer_print_space(p, space);
//  printf("\n");
//  // debug

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

//  // debug
//  isl_printer *p_debug = isl_printer_to_file(sa->ctx, stdout);
//  p_debug = isl_printer_set_yaml_style(p_debug, ISL_YAML_STYLE_BLOCK);
//  p_debug = isl_printer_print_schedule_node(p_debug, node);
//  printf("\n");
//  isl_printer_free(p_debug);
//  // debug

  /* Clean up the band pe_opt properties. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &clear_pe_opt_prop, NULL);

  isl_schedule_free(sa->schedule);
  sa->schedule = schedule;

//  // debug
//  p_debug = isl_printer_to_file(sa->ctx, stdout);
//  p_debug = isl_printer_set_yaml_style(p_debug, ISL_YAML_STYLE_BLOCK);
//  p_debug = isl_printer_print_schedule(p_debug, schedule);
//  printf("\n");
//  isl_printer_free(p_debug);
//  // debug

  /* Clean up. */
  free(tile_size);

  return isl_stat_ok;
}

/* Mark parallel loop as latency_hiding candidate loop. 
 */
static isl_schedule_node *detect_latency_hiding_loop(__isl_take isl_schedule_node *node, void *user)
{
  struct polysa_kernel *sa = user;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      if (isl_schedule_node_band_member_get_coincident(node, i)) {
        node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_latency);
      }
    }
  }

  return node;
}

static isl_bool count_latency_hiding_loop(__isl_keep isl_schedule_node *node, void *user)
{
  int *cnt = user;
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    int n = isl_schedule_node_band_n_member(node);
    for (int i = 0; i < n; i++) {
      if (isl_schedule_node_band_member_get_pe_opt(node, i) == polysa_loop_latency) 
        *cnt = *cnt + 1;
    }
  } 
  
  return isl_bool_true;
}

/* Extract user specified "sa_tile" sizes from the "sa_sizes" command line option,
 * defaulting to option->sa_tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and
 * return a pointer to the tile sizes (or NULL on error).
 * And store the effectively used sizes to sa->used_sizes.
 */
static int *read_latency_tile_sizes(struct polysa_kernel *sa, int *tile_len)
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

/* Given two nested nodes,
 * N1
 * |
 * N2
 * Merge them into one node.
 * N
 * return a pointer to N.
 */
static __isl_give isl_schedule_node *polysa_node_merge(__isl_take isl_schedule_node *node)
{
  if (isl_schedule_node_n_children(node) == 0 || isl_schedule_node_n_children(node) > 1)
    return node;
  isl_schedule_node *parent = node;
  isl_schedule_node *child = isl_schedule_node_child(isl_schedule_node_copy(node), 0);
  if (isl_schedule_node_get_type(parent) != isl_schedule_node_band ||
      isl_schedule_node_get_type(child) != isl_schedule_node_band) 
    return node;

  /* Save the node properties. */
  struct polysa_node_band_prop *parent_prop = extract_node_band_prop(parent);
  struct polysa_node_band_prop *child_prop = extract_node_band_prop(child);

  /* Merge the partial schedules of two nodes. */
  isl_union_pw_aff_list *upa_list = isl_union_pw_aff_list_alloc(
    isl_schedule_node_get_ctx(node), 0);
  isl_space *parent_space = isl_multi_union_pw_aff_get_space(parent_prop->mupa);
  isl_space *child_space = isl_multi_union_pw_aff_get_space(child_prop->mupa);

  for (int i = 0; i < parent_prop->n_member; i++) {
    isl_union_pw_aff *upa = isl_multi_union_pw_aff_get_union_pw_aff(parent_prop->mupa, i);
    upa_list = isl_union_pw_aff_list_add(
      upa_list, upa);
  }
  for (int i = 0; i < child_prop->n_member; i++) {
    isl_union_pw_aff *upa = isl_multi_union_pw_aff_get_union_pw_aff(child_prop->mupa, i);
    upa_list = isl_union_pw_aff_list_add(
      upa_list, upa);
  }

  isl_space *mupa_space = isl_space_add_dims(parent_space, isl_dim_set, isl_space_dim(child_space, isl_dim_set));
  isl_space_free(child_space);

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_print_multi_union_pw_aff(p, parent_prop->mupa);
//  printf("\n");
//  p = isl_printer_print_multi_union_pw_aff(p, child_prop->mupa);
//  printf("\n");
////  p = isl_printer_print_union_pw_aff_list(p, upa_list);
////  printf("\n");
//  p = isl_printer_print_space(p, mupa_space);
//  printf("\n");
//  // debug

  isl_multi_union_pw_aff *mupa = isl_multi_union_pw_aff_from_union_pw_aff_list(
    mupa_space,
    upa_list);

  /* Insert one new node. */
  node = isl_schedule_node_insert_partial_schedule(node, mupa);
  
  /* Restore the node properties. */  
  node = isl_schedule_node_band_set_permutable(node, 1);
  for (int i = 0; i < parent_prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_coincident(node, i, parent_prop->coincident[i]);
  }
  for (int i = 0; i < parent_prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_space_time(node, i, parent_prop->space_time[i]);
    node = isl_schedule_node_band_member_set_pe_opt(node, i, parent_prop->pe_opt[i]);
  }
  for (int i = 0; i < child_prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_coincident(node, i + parent_prop->n_member, child_prop->coincident[i]);
  }
  for (int i = 0; i < child_prop->n_member; i++) {
    node = isl_schedule_node_band_member_set_space_time(node, i + parent_prop->n_member, child_prop->space_time[i]);
    node = isl_schedule_node_band_member_set_pe_opt(node, i + parent_prop->n_member, child_prop->pe_opt[i]);
  }

  /* Delete the old nodes. */
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_delete(node);
  node = isl_schedule_node_delete(node);
  node = isl_schedule_node_parent(node);

  free(parent_prop->coincident);
  free(parent_prop->pe_opt);
  free(parent_prop->space_time);
  isl_multi_union_pw_aff_free(parent_prop->mupa);
  free(parent_prop);

  free(child_prop->coincident);
  free(child_prop->pe_opt);
  free(child_prop->space_time);
  isl_multi_union_pw_aff_free(child_prop->mupa);
  free(child_prop);

  isl_schedule_node_free(child);

//  // debug
//  p = isl_printer_print_multi_union_pw_aff(p, mupa);
//  printf("\n");
//  // debug

  return node;
}

/* Tile the loop at the "pos" position of the band with the size "tile_size".
 * The original band
 * B
 * is first splitted to
 * B1
 * |
 * p
 * |
 * B2
 * The loop p is tiled, and four band nodes are generated.
 * B1
 * |
 * p_tile
 * |
 * B2
 * |
 * p_point
 * The first three bands are then merged together.
 * B'
 * |
 * p_point
 * A pointer to B' is returned.
 */
static __isl_give isl_schedule_node *polysa_node_band_tile_loop(
  __isl_take isl_schedule_node *node, int tile_size, int pos)
{
  isl_multi_val *tile_sizes;
  int n = isl_schedule_node_band_n_member(node);
  int size[1];

  size[0] = tile_size;
  node = isl_schedule_node_band_split(node, pos);
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_band_split(node, 1);
//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  tile_sizes = construct_band_tile_sizes(node, size);
//  // debug
//  p = isl_printer_print_multi_val(p, tile_sizes);
//  printf("\n");
//  // debug
  node = tile_band(node, isl_multi_val_copy(tile_sizes));
  isl_multi_val_free(tile_sizes);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Swap the order of the point band and the next band. */
  node = isl_schedule_node_child(node, 0);
  node = polysa_node_interchange(node);
//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug
    
  /* Merge the first three bands. */
  node = isl_schedule_node_parent(node);
  node = polysa_node_merge(node);
  node = isl_schedule_node_parent(node);
  node = polysa_node_merge(node);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  isl_printer_free(p);
//  // debug

  return node;
}

/* Examine if the node is the last band node, if so, add a "latency" mark before the node. */
static __isl_give isl_schedule_node *add_latency_mark(__isl_take isl_schedule_node *node,
  void *user) {
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
//    // debug
//    isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//    p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
    node = isl_schedule_node_child(node, 0);
    isl_bool no_inner_band = isl_schedule_node_every_descendant(node,
        &no_permutable_node, NULL);
    node = isl_schedule_node_parent(node);
    if (no_inner_band) {
      /* Insert the "latency" mark. */
      isl_id *id = isl_id_alloc(isl_schedule_node_get_ctx(node), "latency", NULL);
//      // debug
//      isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//      p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//      p = isl_printer_print_schedule_node(p, node);
//      printf("\n");
//      // debug
      node = isl_schedule_node_insert_mark(node, id);
    }
  }

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  return node;
}

/* Examine if the node contains any space loops. */
static isl_bool node_has_space(__isl_keep isl_schedule_node *node) {
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return isl_bool_true;

  int n = isl_schedule_node_band_n_member(node);
  for (int i = 0; i < n; i++) {
    if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_space) {
      return isl_bool_true;
    }
  }

  return isl_bool_false;
}

/* Move the current node which represents the latency hiding loop to the below the last time 
 * loop. 
 * If the array is async, then sink the node to the bottom.
 * If the array is sync, then lift it up and insert as the last loop in the time band.
 */
static __isl_give isl_schedule_node *polysa_node_band_sink_time(__isl_take isl_schedule_node *node, struct polysa_kernel *sa) {
  if (sa->type == POLYSA_SA_TYPE_ASYNC) {
    node = isl_schedule_node_band_sink(node);
    /* Add the "latency" mark. */
    node = isl_schedule_node_map_descendant_bottom_up(
      node, &add_latency_mark, NULL);
  } else if (sa->type == POLYSA_SA_TYPE_SYNC) {
    /* Move up to the node that contains the space loop */
//    while(!node_has_space(node)) {
//      node = isl_schedule_node_parent(node);
//    }
    node = isl_schedule_node_parent(node);  

    /* Find the position of the first space loop. */
    int n_member = isl_schedule_node_band_n_member(node);
    int space_pos;
    for (int i = 0; i < n_member; i++) {
      if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_space) {
        space_pos = i;
        break;
      }
    }
    if (space_pos == 0) {
      /* Interchange the current node with the child node. */
      node = polysa_node_interchange(node);
      /* Insert the "latency" mark. */
      isl_id *id = isl_id_alloc(sa->ctx, "latency", NULL);
      node = isl_schedule_node_insert_mark(node, id);
      node = isl_schedule_node_child(node, 0);
      node = isl_schedule_node_child(node, 0);
    } else {
      node = isl_schedule_node_band_split(node, space_pos);
      node = isl_schedule_node_child(node, 0);
      /* Interchange the current node with the child node. */
      node = polysa_node_interchange(node);
      /* Insert the "latency" mark. */
      isl_id *id = isl_id_alloc(sa->ctx, "latency", NULL);
      node = isl_schedule_node_insert_mark(node, id);
      node = isl_schedule_node_child(node, 0);
      node = isl_schedule_node_child(node, 0);
    }
  }

  return node;
}

/* Given each node band, tile the candidate loop and permute it innermost in the time
 * loop band. If the tile size is -1, the candidate loop is skipped.
 * For each point loop, a "latency" mark is added.
 */
static __isl_give isl_schedule_node *polysa_latency_tile_band_loop(
  __isl_take isl_schedule_node *node, void *user)
{
  struct polysa_pe_opt_tile_data *data = user;
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return node;

  int n;
  isl_id *id;
  n = isl_schedule_node_band_n_member(node);

  for (int i = n - 1; i >= 0; i--) {
    if (isl_schedule_node_band_member_get_pe_opt(node, i) == polysa_loop_latency) {      
      int loop_tile_size = data->tile_size[data->tile_len - data->n_touched_loop - 1];
      (data->n_touched_loop)++;
      if (loop_tile_size > 0) {
        /* Tile the current loop and permute it to be the innermost time loop. */

//        // debug
//        isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//        p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//        p = isl_printer_print_schedule_node(p, node);
//        printf("\n");
//        // debug

//        /* Insert the "anchor" mark. */
//        id = isl_id_alloc(data->sa->ctx, "anchor", NULL);
//        node = isl_schedule_node_insert_mark(node, id);
//        node = isl_schedule_node_child(node, 0);

        /* Tile the loop in the band at "i"th position with the size "loop_tile_size".
         * The returned node points at the tile loop. */
        node = polysa_node_band_tile_loop(node, loop_tile_size, i); 

//        // debug
//        p = isl_printer_print_schedule_node(p, node);
//        printf("\n");
//        // debug

        /* Reset the candidate loop in the tile loop the pe_opt property to default. */
        node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
        /* Reset the point loop space_time property to time loop. */
        node = isl_schedule_node_child(node, 0);
        node = isl_schedule_node_band_member_set_space_time(node, 0, polysa_loop_time);
        /* Reset the point loop pe_opt property to default .*/
        node = isl_schedule_node_band_member_set_pe_opt(node, 0, polysa_loop_default);

        /* Move the single loop node to the bottom of the time band. */
        node = polysa_node_band_sink_time(node, data->sa); 
        
//        // debug
//        p = isl_printer_print_schedule_node(p, node);
//        printf("\n");
//        // debug

//        /* Move up to the "anchor" mark. */
//        node = polysa_tree_move_up_to_anchor(node);
//
//        /* Delete the "anchor" mark */
//        node = isl_schedule_node_delete(node);

//        // debug
//        p = isl_printer_print_schedule_node(p, node);
//        printf("\n");
//        // debug

        (data->n_tiled_loop)++;
        return node;
      }
    }
  }

  return node;
}

static __isl_give isl_schedule_node *polysa_latency_tile_loop(__isl_take isl_schedule_node *node, struct polysa_kernel *sa)
{
  /* Count the candidate loop number. */
  int tile_len = 0;
  isl_schedule_node_foreach_descendant_top_down(
      node, &count_latency_hiding_loop, &tile_len);
  // printf("%d\n", tile_len);

  /* Read the tile sizes. */
  int *tile_size;
  tile_size = read_latency_tile_sizes(sa, &tile_len);

  /* Tile the loop. */
  struct polysa_pe_opt_tile_data tile_data = {0, 0, tile_len, tile_size, sa};
  while (tile_data.n_touched_loop != tile_len) {
    node = isl_schedule_node_map_descendant_bottom_up(
      node, &polysa_latency_tile_band_loop, &tile_data);
  }

  free(tile_size);
  return node;
}

// TODO
isl_bool is_innermost_time_loop_parallel(__isl_keep isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    node = isl_schedule_node_child(isl_schedule_node_copy(node), 0);
    isl_bool no_inner_band = isl_schedule_node_every_descendant(node,
        &no_permutable_node, NULL);
    if (!no_inner_band) {
      node = isl_schedule_node_parent(node);
      if (isl_schedule_node_band_member_get_coincident(node, isl_schedule_node_band_n_member(node) - 1) == 0) {
        isl_schedule_node_free(node);
        return isl_bool_false;
      } else {
        isl_schedule_node_free(node);
        return isl_bool_true;
      }
    }
    isl_schedule_node_free(node);
  } 

  return isl_bool_true;
}

/* Apply latency hiding. 
 * Go through all the loops, if there is any parallel loop (considering only RAW), 
 * such a loop will be identified as latency hiding loop candidate. Such loops will be
 * tiled. The point loops will be permuted as the innermost time loops.
 */
isl_stat sa_latency_hiding_optimize(struct polysa_kernel *sa)
{
  isl_schedule *schedule = sa->schedule;
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  
//  /* Detect if the innermost time loop carries RAW dependency. */
//  isl_bool no_opt = isl_schedule_node_every_descendant(
//    node, &is_innermost_time_loop_parallel, NULL);     
  isl_bool no_opt = 0;

  if (!no_opt) {
    printf("[PolySA] Apply latency hiding.\n");
    
    /* Move down to the array marker. */
    node = polysa_tree_move_down_to_array(node, sa->core);
  
    /* Detect all candidate loops. */
    node = isl_schedule_node_map_descendant_bottom_up(
      node, &detect_latency_hiding_loop, sa);
  
    /* Display the candidate loops. */
    isl_schedule_free(schedule);
    schedule = isl_schedule_node_get_schedule(node);
    if (sa->scop->options->debug->polysa_verbose) {
      isl_printer *p = isl_printer_to_file(sa->ctx, stdout);
      p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
      p = isl_printer_print_schedule(p, schedule);
      printf("\n");
      isl_printer_free(p);
    }
    isl_schedule_free(schedule);
  
    /* Tile the candidate loop. 
     * For each candidate loop, if the loop is used for latency hiding,
     * it is tiled and permuted to the innermost of the time loop band. 
     * A latency hiding marker is added. */
    node = polysa_latency_tile_loop(node, sa);
  
    /* Clean up the band pe_opt properties. */
    schedule = isl_schedule_node_get_schedule(node);
    isl_schedule_node_free(node);
    schedule = isl_schedule_map_schedule_node_bottom_up(
        schedule, &clear_pe_opt_prop, NULL);
  
    sa->schedule = schedule;
  } else {
    isl_schedule_node_free(node);
  }

  return isl_stat_ok;
}

/* Apply PE optimization including:
 * - latency hiding
 * - SIMD vectorization
 * - array partitioning
 */
isl_stat sa_pe_optimize(struct polysa_kernel *sa, bool pass_en[])
{
  /* Prepartion before starting the optimization. */
  /* Initialize the polysa_loop_types. */
  sa_loop_init(sa);
  /* Set up the space_time properties. */
  sa_space_time_loop_setup(sa);
  /* Extract the tile sizes. */
  sa->sizes = extract_sizes_from_str(sa->ctx, sa->scop->options->sa_sizes);
  /* Set the kernel id. */
  sa->id = 0;
  /* Set the core */
  isl_schedule_node *root = isl_schedule_get_root(sa->schedule);
  isl_union_set *domain = isl_schedule_node_get_domain(root);
  sa->core = isl_union_set_universe(domain);
  isl_schedule_node_free(root);

  /* Array partitioning. */
  if (pass_en[0])
    sa_array_partitioning_optimize(sa);
  /* Latency hiding. */
  if (pass_en[1])
    sa_latency_hiding_optimize(sa);
  /* SIMD vectorization. */
  if (pass_en[2])
    sa_SIMD_vectorization_optimize(sa);
}

/* Apply SIMD vectorization. 
 * Go through all the loops, if there is any vectorizable loop (parallel or reduction loop
 * with stride-0/1 access), such a loop will be identified as SIMD loop candidate. We will rank
 * the loops by heuristics and pick up one loop to be tiled. The point loops will be permuated 
 * as the innermost loops to be unrolled.
 */
isl_stat sa_SIMD_vectorization_optimize(struct polysa_kernel *sa)
{
  printf("[PolySA] Apply SIMD vectorization.\n");
  isl_schedule *schedule = sa->schedule; 

  return isl_stat_ok;
}

/* Initialize the space_time and pe_opt to polysa_loop_default for all band nodes. */
static __isl_give isl_schedule_node *init_band_node_sa_properties(__isl_take isl_schedule_node *node, void *user) 
{
  if (!node)
    return NULL;

  struct polysa_kernel *sa = user;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    int band_w = isl_schedule_node_band_n_member(node);
    /* Initialize the SA properties. */
    for (int i = 0; i < band_w; i++) {
      node = isl_schedule_node_band_member_set_space_time(node, i, polysa_loop_time);
      node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
    }
  }

  return node;
}

/* Initialize the fields of time_space and pe_opt for each band node in the schedule tree. */
isl_stat sa_loop_init(struct polysa_kernel *sa)
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

/* Set up the space_time properties. As all the loops are initialized to be the time loop,
 * only the space loops are to be set.
 */
isl_stat sa_space_time_loop_setup(struct polysa_kernel *sa) 
{
  isl_schedule_node *node;
  if (sa->type == POLYSA_SA_TYPE_SYNC) {
    node = get_innermost_permutable_node(sa->schedule);
    for (int i = isl_schedule_node_band_n_member(node) - sa->space_w; i < isl_schedule_node_band_n_member(node); i++) {
      node = isl_schedule_node_band_member_set_space_time(node, i, polysa_loop_space);      
    }
  } else if (sa->type == POLYSA_SA_TYPE_ASYNC) {
    node = get_outermost_permutable_node(sa->schedule);
    for (int i = 0; i < sa->space_w; i++) {
      node = isl_schedule_node_band_member_set_space_time(node, i, polysa_loop_space);
    }
  }

  isl_schedule *schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  isl_schedule_free(sa->schedule);
  sa->schedule = schedule;

  return isl_stat_ok;
}

/* Apply space-time transformation to generate different systolic array candidates. */
struct polysa_kernel **sa_space_time_transform(__isl_take isl_schedule *schedule, 
    struct ppcg_scop *scop, isl_size *num_sa) 
{
  struct polysa_kernel **sa_list = NULL;
  isl_size n_sa = 0;

  isl_schedule_node *band = get_outermost_permutable_node(schedule);
  isl_size band_w = isl_schedule_node_band_n_member(band); 
  /* Explore 1D systolic array */
  if (scop->options->max_sa_dim >= 1 && band_w >= 1) {
    printf("[PolySA] Explore 1D systolic array.\n");
    isl_size n_sa_dim = 0;
    struct polysa_kernel **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 1, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_kernel **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_kernel *));
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
    struct polysa_kernel **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 2, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_kernel **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_kernel *));
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
    struct polysa_kernel **sa_dim_list = sa_space_time_transform_at_dim(schedule, scop, 3, &n_sa_dim);
    printf("[PolySA] %d candidates generated.\n", n_sa_dim);
    sa_list = (struct polysa_kernel **)realloc(sa_list, (n_sa + n_sa_dim) * sizeof(struct polysa_kernel *));
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
struct polysa_kernel *sa_candidates_smart_pick(struct polysa_kernel **sa_list, __isl_keep isl_size num_sa)
{
  assert(num_sa > 0);
  struct polysa_kernel *sa_opt = polysa_kernel_copy(sa_list[3]);
    
  for (int i = 0; i < num_sa; i++)
    polysa_kernel_free(sa_list[i]);
  free(sa_list);

  return sa_opt;
}

/* Generate syncrhonized systolic arrays with the given dimension.
 * For sync arrays, time loops are placed outside the space loops.
 */
struct polysa_kernel **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa)
{
  struct polysa_kernel **sas = NULL;  

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
        struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
        sa->scop = scop;
        sa->type = POLYSA_SA_TYPE_SYNC;

        /* Update the array dimension. */
        sa->array_dim = dim;
        sa->array_part_w = 0;
        sa->space_w = dim;
        sa->time_w = band_w - dim;

        /* Add the new variant into the list. */
        sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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
            struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
            sa->scop = scop;
            sa->type = POLYSA_SA_TYPE_SYNC;

            /* Update the array dimension. */
            sa->array_dim = dim;
            sa->array_part_w = 0;
            sa->space_w = dim;
            sa->time_w = band_w - dim;

            /* Add the new variant into the list. */
            sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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
                struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
                sa->scop = scop;
                sa->type = POLYSA_SA_TYPE_SYNC;

                /* Update the array dimension. */
                sa->array_dim = dim;
                sa->array_part_w = 0;
                sa->space_w = dim;
                sa->time_w = band_w - dim;
    
                /* Add the new variant into the list. */
                sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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

struct polysa_kernel **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, 
    isl_size dim, isl_size *num_sa)
{
  if (scop->options->sa_type == POLYSA_SA_TYPE_ASYNC) {
    return sa_space_time_transform_at_dim_async(schedule, scop, dim, num_sa);
  } else if (scop->options->sa_type == POLYSA_SA_TYPE_SYNC) {
    return sa_space_time_transform_at_dim_sync(schedule, scop, dim, num_sa);
  }
}

/* Generate asynchronized systolic arrays with the given dimension. 
 * For async arrays, space loops are placed outside the time loops.
 */
struct polysa_kernel **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa) 
{
  struct polysa_kernel **sas = NULL;  

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
        struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
        sa->scop = scop;
        sa->type = POLYSA_SA_TYPE_ASYNC;

        /* Update the array dimension. */
        sa->array_dim = dim;
        sa->array_part_w = 0;
        sa->space_w = dim;
        sa->time_w = band_w - dim;

        /* Add the new variant into the list. */
        sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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
            struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
            sa->scop = scop;
            sa->type = POLYSA_SA_TYPE_ASYNC;

            /* Update the array dimension. */
            sa->array_dim = dim;
            sa->array_part_w = 0;
            sa->space_w = dim;
            sa->time_w = band_w - dim;

            /* Add the new variant into the list. */
            sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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
                struct polysa_kernel *sa = polysa_kernel_from_schedule(new_schedule);
                sa->scop = scop;
                sa->type = POLYSA_SA_TYPE_ASYNC;

                /* Update the array dimension. */
                sa->array_dim = dim;
                sa->array_part_w = 0;
                sa->space_w = dim;
                sa->time_w = band_w - dim;
    
                /* Add the new variant into the list. */
                sas = (struct polysa_kernel **)realloc(sas, (*num_sa + 1) * sizeof(struct polysa_kernel *));
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

/* A program is legal to be transformed to systolic array if and on if 
 * it satisfies the following constraints:
 * - one single fully permutable outermost band
 * - uniform dependency
 */
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop) 
{
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

/* Create the array of polysa_local_array_info structures "array"
 * inside "kernel". The number of elements in this array is 
 * the same as the number of arrays in "prog".
 * Initialize the "array" field of each local array to point 
 * to the corresponding array in "prog".
 */
static struct polysa_kernel *polysa_kernel_create_local_arrays(
    struct polysa_kernel *kernel, struct polysa_prog *prog)
{
  int i;
  isl_ctx *ctx;

  if (!kernel)
    return NULL;

  ctx = isl_set_get_ctx(prog->context);
  kernel->array = isl_calloc_array(ctx,
      struct polysa_local_array_info, prog->n_array);
  if (!kernel->array)
    return polysa_kernel_free(kernel);
  kernel->n_array = prog->n_array;

  for (i = 0; i < prog->n_array; i++) 
    kernel->array[i].array = &prog->array[i];

  return kernel;
}

/* Mark all dimensions in the current band node atomic.
 */
static __isl_give isl_schedule_node *atomic(__isl_take isl_schedule_node *node)
{
	return ppcg_set_schedule_node_type(node, isl_ast_loop_atomic);
}

/* Mark "node" atomic, if it is a band node.
 * Do the same for all ancestors.
 * Return a pointer to "node" (in the updated schedule tree).
 */
static __isl_give isl_schedule_node *atomic_ancestors(
	__isl_take isl_schedule_node *node)
{
	int pos;

	if (!node)
		return NULL;
	if (!isl_schedule_node_has_parent(node))
		return node;

	pos = isl_schedule_node_get_child_position(node);
	node = isl_schedule_node_parent(node);
	if (isl_schedule_node_get_type(node) == isl_schedule_node_band)
		node = atomic(node);
	node = atomic_ancestors(node);
	node = isl_schedule_node_child(node, pos);

	return node;
}

/* Wrapper around polysa_kernel_free for use as a isl_id_set_free_user callback.
 */
static void polysa_kernel_free_wrap(void *user) 
{
  struct polysa_kernel *kernel = user;

  polysa_kernel_free(kernel);
}

/* Group the domain elements into a single space, named kernelX,
 * with X the kernel sequence number "kernel_id".
 */
static __isl_give isl_schedule_node *group_statements(
	__isl_take isl_schedule_node *node, int kernel_id)
{
	char buffer[20];
	isl_id *id;

	if (!node)
		return NULL;

	snprintf(buffer, sizeof(buffer), "kernel%d", kernel_id);
	id = isl_id_alloc(isl_schedule_node_get_ctx(node), buffer, NULL);
	return isl_schedule_node_group(node, id);
}

/* Extract the set of parameter values and outer schedule dimensions
 * for which any statement instance
 * in the kernel inserted at "node" needs to be executed.
 * Intersect the set of parameter values derived from the host schedule
 * relation with the context of "prog".
 */
static __isl_give isl_set *extract_context(__isl_keep isl_schedule_node *node,
	struct polysa_prog *prog)
{
	isl_union_map *schedule;
	isl_union_set *schedule_domain;
	isl_set *context;
	int empty;

	schedule = isl_schedule_node_get_prefix_schedule_relation(node);
	schedule_domain = isl_union_map_range(schedule);
	empty = isl_union_set_is_empty(schedule_domain);
	if (empty < 0) {
		isl_union_set_free(schedule_domain);
		return NULL;
	}
	if (empty) {
		int depth;
		isl_space *space;

		space = isl_union_set_get_space(schedule_domain);
		isl_union_set_free(schedule_domain);
		space = isl_space_set_from_params(space);
		depth = isl_schedule_node_get_schedule_depth(node);
		space = isl_space_add_dims(space, isl_dim_set, depth);
		context = isl_set_empty(space);
	} else {
		context = isl_set_from_union_set(schedule_domain);
	}
	context = isl_set_intersect_params(context,
					    isl_set_copy(prog->context));

	return context;
}

/* Return the set of outer array elements accessed by
 * by the statement instances in "domain" in "prog".
 * The instances in "domain" are those that appear
 * in the domains of the access relations in "prog".
 */
static __isl_give isl_union_set *accessed_by_domain(
	__isl_take isl_union_set *domain, struct polysa_prog *prog)
{
	isl_union_map *access;
	isl_union_set *arrays;

	access = isl_union_map_union(isl_union_map_copy(prog->read),
				     isl_union_map_copy(prog->may_write));
	access = isl_union_map_intersect_domain(access, domain);
	arrays = isl_union_map_range(access);
	arrays = isl_union_set_apply(arrays,
				isl_union_map_copy(prog->to_outer));

	return arrays;
}

/* Replace "pa" by the zero function defined over the universe domain
 * in the space of "pa".
 */
static __isl_give isl_pw_aff *set_universally_zero(__isl_take isl_pw_aff *pa)
{
	isl_space *space;
	isl_aff *zero;

	space = isl_space_domain(isl_pw_aff_get_space(pa));
	isl_pw_aff_free(pa);
	zero = isl_aff_zero_on_domain(isl_local_space_from_space(space));

	return isl_pw_aff_from_aff(zero);
}

/* The sizes of the arrays on the host that have been computed by
 * extract_array_info may depend on the parameters.  Use the extra
 * constraints on the parameters that are valid at "host_domain"
 * to simplify these expressions and store the results in kernel->array.
 *
 * We only need these localized bounds for arrays that are accessed
 * by the current kernel.  If we have found at least one reference group
 * then the array is accessed by the kernel.
 *
 * The resulting sizes may be functions that are nowhere defined
 * in case the access function cannot possibly access anything inside
 * the kernel for some reason.  If so, they are replaced by the zero
 * function.  Since the access function cannot actually access anything,
 * there is no harm in printing the array sizes as zero.
 */
static void localize_bounds(struct polysa_kernel *kernel,
	__isl_keep isl_set *host_domain)
{
	int i, j;
	isl_set *context;

	context = isl_set_copy(host_domain);
	context = isl_set_params(context);

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *local = &kernel->array[i];
		isl_multi_pw_aff *bound;
		int n_index;

		if (local->n_group == 0)
			continue;

		n_index = local->array->n_index;
		bound = isl_multi_pw_aff_copy(local->array->bound);

		for (j = 0; j < n_index; ++j) {
			isl_pw_aff *pwaff;
			int empty;

			pwaff = isl_multi_pw_aff_get_pw_aff(bound, j);
			pwaff = isl_pw_aff_gist(pwaff, isl_set_copy(context));
			empty = isl_pw_aff_is_empty(pwaff);
			if (empty < 0)
				pwaff = isl_pw_aff_free(pwaff);
			else if (empty)
				pwaff = set_universally_zero(pwaff);
			bound = isl_multi_pw_aff_set_pw_aff(bound, j, pwaff);
		}

		local->n_index = n_index;
		local->bound = bound;
	}
	isl_set_free(context);
}

/* If max_local_memory is not set to infinity (-1), then make
 * sure that the total amount of local memory required by the
 * array reference groups mapped to local memory by "kernel"
 * is no larger than this maximum.
 *
 * We apply a greedy approach and discard (keep in global memory)
 * those groups that would result in a total memory size that
 * is larger than the maximum.
 *
 * This function should be called after any function that may
 * affect the decision on whether to place a reference group
 * in local or global memory.
 */
static void check_local_memory_bound(struct polysa_kernel *kernel)
{
	int i, j;
	isl_val *left, *size;

	if (kernel->options->max_local_memory < 0)
		return;

	left = isl_val_int_from_si(kernel->ctx,
				    kernel->options->max_local_memory);

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *local = &kernel->array[i];

		for (j = 0; j < local->n_group; ++j) {
			struct polysa_array_ref_group *group;
			enum polysa_group_access_type type;

			group = local->groups[j];
			type = polysa_array_ref_group_type(group);
			if (type != POLYSA_ACCESS_LOCAL)
				continue;

			size = polysa_array_tile_size(group->local_tile);
			size = isl_val_mul_ui(size, local->array->size); // Byte

			if (isl_val_le(size, left)) {
				left = isl_val_sub(left, size);
				continue;
			}
			isl_val_free(size);

			group->local_tile =
					polysa_array_tile_free(group->local_tile);
		}
	}

	isl_val_free(left);
}

/* Mark all arrays of "kernel" that have an array reference group
 * that is not mapped to local memory as
 * accessing the corresponding global device memory.
 */
static void mark_global_arrays(struct polysa_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *local = &kernel->array[i];

		if (local->global)
			continue;
		for (j = 0; j < local->n_group; ++j) {
			if (polysa_array_ref_group_tile(local->groups[j]))
				continue;

			local->global = 1;
			local->array->global = 1;
			break;
		}
	}
}

/* Compute a tiling for all the array reference groups in "kernel".
 */
static void compute_group_tilings(struct polysa_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j)
			polysa_array_ref_group_compute_tiling(array->groups[j]);
	}
}

/* Return the union of all tagged access relations in the group.
 */
static __isl_give isl_union_map *group_tagged_access_relation(
	struct polysa_array_ref_group *group)
{
	int i;
	isl_union_map *access;

	access = isl_union_map_empty(isl_map_get_space(group->access));
	for (i = 0; i < group->n_ref; ++i) {
		isl_map *map_i;

		map_i = isl_map_copy(group->refs[i]->tagged_access);
		access = isl_union_map_union(access,
					    isl_union_map_from_map(map_i));
	}

	return access;
}

/* Given an access relation "access" from "group", remove those reads
 * if ("read" is 1) or writes (if "read" is 0) that are only needed to
 * communicate data within the same iteration of the schedule "prefix"
 * at the position where the copying of the group is inserted.
 * That is, the output dimension of "prefix"
 * is equal to tile->depth.
 * The domain of "prefix" corresponds to the original statement instances,
 * i.e., those that appear in the domains of the access relations.
 *
 * Extract the tagged access relation of "group" and
 * then call remove_local_accesses.
 */
static __isl_give isl_union_map *remove_local_accesses_group(
	struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
	__isl_take isl_union_map *access, __isl_keep isl_union_map *prefix,
	int read)
{
	isl_union_map *sched, *tagged;

	if (isl_union_map_is_empty(access))
		return access;

	tagged = group_tagged_access_relation(group);
	sched = isl_union_map_copy(prefix);

	return remove_local_accesses(kernel->prog, tagged, access, sched, read);
}

/* Return a read ("read" is 1) or write access relation for "group"
 * with those accesses removed that are only needed to communicate data
 * within the subtree of the schedule rooted at "node".
 * Furthermore, include the prefix schedule at "node".
 * That is, return a relation of the form
 *
 *	S -> [D -> A]
 *
 * with D the outer schedule dimensions at "node".
 */
static __isl_give isl_union_map *anchored_non_local_accesses(
	struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	isl_union_map *access;
	isl_union_map *prefix;

	prefix = isl_schedule_node_get_prefix_schedule_relation(node);
	prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
			    isl_union_pw_multi_aff_copy(kernel->contraction));
	access = polysa_array_ref_group_access_relation(group, read, !read);
	access = remove_local_accesses_group(kernel, group, access, prefix,
						read);
  /* Prefix: S -> D
   * Access: S -> A
   * range_product: S -> [D -> A]
   */
	access = isl_union_map_range_product(prefix, access);

	return access;
}

/* Given an array reference group "group", create a mapping
 *
 *	read[D -> A] -> [D -> A]
 *
 * if "read" is set or
 *
 *	write[D -> A] -> [D -> A]
 *
 * if "read" is not set.
 * D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 */
static __isl_give isl_multi_aff *create_from_access(isl_ctx *ctx,
	struct polysa_array_ref_group *group, int read)
{
	struct polysa_array_tile *tile;
	isl_space *space;
	isl_id *id;

	tile = polysa_array_ref_group_tile(group);
	space = isl_space_copy(group->array->space);
	space = isl_space_from_range(space);
	space = isl_space_add_dims(space, isl_dim_in, tile->depth);
	space = isl_space_wrap(space);
	space = isl_space_map_from_set(space);

	id = isl_id_alloc(ctx, read ? "read" : "write", group);
	space = isl_space_set_tuple_id(space, isl_dim_in, id);

	return isl_multi_aff_identity(space);
}

/* Return the extent of "array", recomputed from the bounds.
 * The recomputed extent may be simpler than the original extent.
 */
static __isl_give isl_set *array_extent(struct polysa_array_info *array)
{
	int i;
	isl_id *id;
	isl_space *space;
	isl_local_space *ls;
	isl_set *extent;

	id = isl_set_get_tuple_id(array->extent);
	space = isl_set_get_space(array->extent);
	extent = isl_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);
	for (i = 0; i < array->n_index; ++i) {
		isl_pw_aff *bound;
		isl_aff *aff;
		isl_pw_aff *index;
		isl_set *lt;

		extent = isl_set_lower_bound_si(extent, isl_dim_set, i, 0);

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, i);
		index = isl_pw_aff_from_aff(aff);
		bound = isl_multi_pw_aff_get_pw_aff(array->bound, i);
		bound = isl_pw_aff_from_range(bound);
		bound = isl_pw_aff_add_dims(bound, isl_dim_in, array->n_index);
		bound = isl_pw_aff_set_tuple_id(bound, isl_dim_in,
						isl_id_copy(id));
		lt = isl_pw_aff_lt_set(index, bound);
		extent = isl_set_intersect(extent, lt);
	}
	isl_local_space_free(ls);
	isl_id_free(id);

	return extent;
}

/* Return a map from the first group->local_tile->depth dimensions
 * of the computed schedule to the array tile in
 * global memory that corresponds to the local memory copy.
 *
 * In particular, return a map
 *
 *	{ D[i] -> A[a] }
 *
 * with constraints
 *
 *	tile_offset(i) <= a <= tile_offset(i) + tile_size - 1		(1)
 *
 * and
 *
 *	0 <= a <= array_size - 1					(2)
 *
 * Note that if some stride has been detected (i.e., when
 * group->local_tile->bound[i].shift is set), then a in (1) refers
 * to the shifted and scaled down version.
 *
 * Constraints (1) are obtained by mapping the size constraints on the
 * local memory tile back to the access relation.
 * Constraints (2) are obtained from the (recomputed) extent.
 */
static __isl_give isl_map *group_tile(struct polysa_array_ref_group *group)
{
	int i;
	int n_index = group->array->n_index;
	isl_map *tile;
	isl_space *space;
	isl_set *local;
	isl_set *extent;

	space = isl_multi_aff_get_space(group->local_tile->tiling);
	space = isl_space_range(space);
	local = isl_set_universe(space);
	for (i = 0; i < n_index; ++i) {
		isl_val *bound;

		local = isl_set_lower_bound_si(local, isl_dim_set, i, 0);
		bound = isl_val_copy(group->local_tile->bound[i].size);
		bound = isl_val_sub_ui(bound, 1);
		local = isl_set_upper_bound_val(local, isl_dim_set, i, bound);
	}
	local = isl_set_preimage_multi_aff(local,
				isl_multi_aff_copy(group->local_tile->tiling));
	tile = isl_set_unwrap(local);
	extent = array_extent(group->array);
	tile = isl_map_intersect_range(tile, extent);

	return tile;
}

/* Add copy statements to the schedule tree of "node"
 * for reading from global memory to local memory (if "read" is set) or
 * for writing back from local memory to global memory
 * (if "read" is not set) for the array reference group "group" that
 * is mapped to local memory.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 *
 * The copies are performed in the order of the corresponding local
 * memory tile.
 * The copy statement instances include a reference to the outer
 * tile->depth dimensions of the kernel schedule for ease of
 * combining them with the group tiling.
 *
 * If we are performing a read from global memory to local memory and
 * if the array involved is not a scalar, then we copy
 * the entire tile to local memory.  This may result in some extra
 * elements getting copied, but it should lead to simpler code
 * (which means that fewer registers may be needed) and less divergence.
 *
 * Otherwise, we only copy the elements that will be read or have been written
 * in the kernel.
 *
 * That is, the extra schedule is of the form
 *
 *	type[D -> A] -> T
 *
 * where D corresponds to the outer tile->depth dimensions of
 * the kernel schedule, A to the global array and T is the corresponding
 * local memory tile.
 *
 * The copying is inserted in the schedule tree through an extension
 * of the form
 *
 *	D -> type[D -> A]
 *
 * where the extra domain elements type[D -> A] are those accessed
 * by the group.  In the case of read from a non-scalar, this set
 * is replaced by the entire local memory tile.
 *
 * If the "unroll_copy_local" option is set, then the AST generator
 * is instructed to unroll the copying code.
 *
 * The extension is inserted before the core computation in case of a read
 * and after the core computation in case of a write.
 */
/*
 * A filter is inserted on type[D -> A] to map the copy instances
 * to the threads.  In particular, the thread identifiers are
 * equated to the position inside the shared memory tile (T)
 * modulo the block size.
 * We try to align the innermost tile dimension with the innermost
 * thread identifier (x) as a heuristic to improve coalescing.
 * In particular, if the dimension of the tile is greater than
 * the dimension of the block, then the schedule mapping to the tile
 * is broken up into two pieces and the filter is applied to the inner part.
 * If, on the other hand, the dimension of the tile is smaller than
 * the dimension of the block, then the initial thread identifiers
 * are equated to zero and the remaining thread identifiers are
 * matched to the memory tile.
 *
 * In the case of a read, we first need to make sure there is some
 * synchronization before the core computation such that we can put the read
 * from global memory to shared memory before that synchronization.
 * This ensures that all threads have finished copying into shared memory
 * before the shared memory is used.
 * We also need to make sure that there is a synchronization node after
 * the core computation to ensure that the next load into shared memory
 * only happens after all data has been used.  There is no need for
 * this synchronization if we are at the outer level since then there
 * won't be a next load.
 * In the case of a write, we need to make sure there is some synchronization
 * after the core computation such that we can put the write from shared
 * memory to global memory after that synchronization.
 * Unless we are at the outer level, we also need a synchronization node
 * after the write to ensure the data is saved to global memory
 * before the next iteration writes to the same shared memory.
 * It also makes sure the data has arrived in global memory before
 * it is read in a subsequent iteration.
 */
static __isl_give isl_schedule_node *add_copies_group_local(
	struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	struct polysa_array_tile *tile;
	isl_union_map *access;
	isl_union_set *domain;
	isl_multi_aff *ma;
	isl_multi_aff *from_access;
	isl_multi_pw_aff *mpa;
	isl_multi_union_pw_aff *mupa;
	isl_schedule_node *graft;
	isl_union_set *filter;
	int skip;
	int kernel_depth;
	int empty;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

	tile = polysa_array_ref_group_tile(group);
	kernel_depth = isl_schedule_node_get_schedule_depth(node);
	node = polysa_tree_move_down_to_depth(node, tile->depth, kernel->core);

  /* S -> [D -> A] */
	access = anchored_non_local_accesses(kernel, group, node, read); 
	empty = isl_union_map_is_empty(access);
	if (empty < 0 || empty) {
		isl_union_map_free(access);
		if (empty < 0)
			return isl_schedule_node_free(node);
		return polysa_tree_move_up_to_kernel(node);
	}

	group->array->global = 1;
	group->local_array->global = 1;

  /* read[D -> A] -> [D -> A] */
	from_access = create_from_access(kernel->ctx, group, read); 

  /* [D -> A] -> T */
	ma = isl_multi_aff_copy(tile->tiling);
	ma = isl_multi_aff_pullback_multi_aff(ma,
					    isl_multi_aff_copy(from_access));
	mpa = isl_multi_pw_aff_from_multi_aff(ma);
  /* read[D -> A] -> T */
	mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

  /* [D -> A] */
	domain = isl_union_map_range(access);

	if (read && !polysa_array_is_scalar(group->array)) {
		isl_map *map;
		isl_union_set_free(domain);
		map = group_tile(group);
		domain = isl_union_set_from_set(isl_map_wrap(map));
	}

  /* read[D -> A] */
	domain = isl_union_set_preimage_multi_aff(domain, from_access);
  /* read[D -> A] -> D */
	access = isl_union_set_wrapped_domain_map(domain);
  /* D -> read[D -> A] */
	access = isl_union_map_reverse(access);
	access = isl_union_map_coalesce(access);
	graft = isl_schedule_node_from_extension(access);

	graft = isl_schedule_node_child(graft, 0);

	graft = isl_schedule_node_insert_partial_schedule(graft, mupa); 
	if (kernel->options->unroll_copy_local)
		graft = ppcg_set_schedule_node_type(graft, isl_ast_loop_unroll);

//	if (tile->n > kernel->n_block && kernel->n_block > 0) {
//		graft = isl_schedule_node_band_split(graft,
//						tile->n - kernel->n_block);
//		graft = isl_schedule_node_child(graft, 0);
//	}
//	if (tile->n < kernel->n_block)
//		skip = kernel->n_block - tile->n;
//	else
//		skip = 0;
//	filter = set_schedule_modulo(graft, kernel->thread_ids,
//					kernel->block_dim);
//	if (!kernel->options->wrap)
//		graft = snap_band_to_sizes(graft, kernel->block_dim + skip,
//			    kernel->options);
//	if (tile->n > kernel->n_block && kernel->n_block > 0)
//		graft = isl_schedule_node_parent(graft);
//	graft = isl_schedule_node_insert_filter(graft, filter);

	while (graft && isl_schedule_node_has_parent(graft))
		graft = isl_schedule_node_parent(graft);

	if (read) {
//		if (kernel_depth < tile->depth)
//			node = gpu_tree_ensure_sync_after_core(node, kernel);
//		node = gpu_tree_move_left_to_sync(node, kernel);
		node = isl_schedule_node_graft_before(node, graft);
	} else {
//		node = gpu_tree_move_right_to_sync(node, kernel);
		node = isl_schedule_node_graft_after(node, graft);
//		if (kernel_depth < tile->depth)
//			node = add_group_write_sync(node, kernel, group, 1);
	}

	node = polysa_tree_move_up_to_kernel(node);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

	return node;
}

/* Check whether the array reference group "group" is mapped to
 * private or shared memory and, if so,
 * add copy statements to the schedule tree of "node"
 * for reading from global memory to private or shared memory
 * (if "read" is set) or for writing back from private or shared memory
 * to global memory (if "read" is not set) for this group.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 */
static __isl_give isl_schedule_node *add_copies_group(
	struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	enum polysa_group_access_type type;

	type = polysa_array_ref_group_type(group);
	if (type == POLYSA_ACCESS_LOCAL)
		return add_copies_group_local(kernel, group, node, read); 
//	if (type == ppcg_access_shared)
//		return add_copies_group_shared(kernel, group, node, read);
	return node;
}

/* For each array reference group that is mapped to local memory,
 * add copy statements to the schedule tree of "node"
 * for reading from global memory to private or shared memory
 * and for writing back.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 */
static __isl_give isl_schedule_node *add_copies(struct polysa_kernel *kernel,
	__isl_take isl_schedule_node *node)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct polysa_array_ref_group *group = array->groups[j];

			node = add_copies_group(kernel, group, node, 1);
			if (!node)
				return NULL;
			node = add_copies_group(kernel, group, node, 0);
			if (!node)
				return NULL;
		}
	}

	return node;
}

static void create_kernel_var(isl_ctx *ctx, struct polysa_array_ref_group *group,
	struct polysa_kernel_var *var)
{
	int j;
	struct polysa_array_tile *tile;
	isl_printer *p;

	var->array = group->array;

	var->type = polysa_array_ref_group_type(group);
	tile = polysa_array_ref_group_tile(group);

	p = isl_printer_to_str(ctx);
	p = polysa_array_ref_group_print_name(group, p);
	var->name = isl_printer_get_str(p);
	isl_printer_free(p);

	var->size = isl_vec_alloc(ctx, group->array->n_index);

	for (j = 0; j < group->array->n_index; ++j)
		var->size = isl_vec_set_element_val(var->size, j,
					    isl_val_copy(tile->bound[j].size));
}

static isl_stat create_kernel_vars(struct polysa_kernel *kernel)
{
	int i, j, n;

	n = 0;
	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct polysa_array_ref_group *group = array->groups[j];
			enum polysa_group_access_type type;

			type = polysa_array_ref_group_type(group);
			if (type != POLYSA_ACCESS_GLOBAL)
				++n;
		}
	}

	kernel->var = isl_calloc_array(kernel->ctx, struct polysa_kernel_var, n);
	if (!kernel->var)
		return isl_stat_error;
	kernel->n_var = n;

	n = 0;
	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct polysa_array_ref_group *group = array->groups[j];
			enum polysa_group_access_type type;

			type = polysa_array_ref_group_type(group);
			if (type == POLYSA_ACCESS_GLOBAL)
				continue;
			create_kernel_var(kernel->ctx, group, &kernel->var[n]);
			++n;
		}
	}

	return isl_stat_ok;
}

/* Compute the effective grid size as a list of the sizes in each dimension.
 *
 * The grid size specified by the user or set by default
 * in read_grid_sizes() and applied by the block filter,
 * may be too large for the given code in the sense that
 * it may contain blocks that don't need to execute anything.
 * We therefore don't return this grid size, but instead the
 * smallest grid size that ensures that all blocks that actually
 * execute code are included in the grid.
 *
 * We first extract a description of the grid, i.e., the possible values
 * of the block ids, from the domain elements in "domain" and
 * kernel->block_filter.
 * The block ids are parameters in kernel->block_filter.
 * We simply need to change them into set dimensions.
 *
 * Then, for each block dimension, we compute the maximal value of the block id
 * and add one.
 */
static __isl_give isl_multi_pw_aff *extract_grid_size(
	struct polysa_kernel *kernel, __isl_take isl_union_set *domain)
{
	int i;
	isl_set *grid;
	isl_set *context;
	isl_multi_pw_aff *size;

  /* For PolySA, we set the grid size as 1 */
  grid = isl_union_set_params(domain);
  grid = isl_set_from_params(grid);
  grid = isl_set_add_dims(grid, isl_dim_set, kernel->n_grid);
  for (i = 0; i < kernel->n_grid; ++i) {
    int pos;
    isl_constraint *ls;

    if (!grid)
      return NULL;

    /* Set this dimension as 1. */
    ls = isl_constraint_alloc_equality(isl_local_space_from_space(isl_set_get_space(grid)));
    ls = isl_constraint_set_constant_si(ls, 0);
    ls = isl_constraint_set_coefficient_si(ls, isl_dim_set, i, 1);
    grid = isl_set_add_constraint(grid, ls);
  }

  grid = isl_set_coalesce(grid);
  size = ppcg_size_from_extent(grid);
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_multi_pw_aff_get_ctx(size), stdout);
//  p = isl_printer_print_multi_pw_aff(p, size);
//  printf("\n");
//  // debug
  context = isl_set_params(isl_set_copy(kernel->context));
  return isl_multi_pw_aff_gist(size, context);

//	domain = isl_union_set_intersect(domain,
//				    isl_union_set_copy(kernel->block_filter));
//  
//	grid = isl_union_set_params(domain);
//	grid = isl_set_from_params(grid);
//	grid = isl_set_add_dims(grid, isl_dim_set, kernel->n_grid);
//
//	for (i = 0; i < kernel->n_grid; ++i) {
//		int pos;
//		isl_id *id;
//
//		if (!grid)
//			return NULL;
//
//		id = isl_id_list_get_id(kernel->block_ids, i);
//		pos = isl_set_find_dim_by_id(grid, isl_dim_param, id);
//		isl_id_free(id);
//		if (pos < 0)
//			isl_die(isl_set_get_ctx(grid), isl_error_internal,
//				"missing constraints on block identifier",
//				grid = isl_set_free(grid));
//		grid = isl_set_equate(grid, isl_dim_param, pos, isl_dim_set, i);
//		grid = isl_set_project_out(grid, isl_dim_param, pos, 1);
//	}
//
//	grid = isl_set_coalesce(grid);
//	size = ppcg_size_from_extent(grid);
//	context = isl_set_params(isl_set_copy(kernel->context));
//	return isl_multi_pw_aff_gist(size, context);
}

/* Create polysa_kernel represents the domain isntances that reach "node" and 
 * insert a mark node pointing to the polyhedral_kernel before "node".
 *
 * Mark all outer band nodes as atomic to ensure each kernel is only scheduled once.
 * If the domain elements that reach "node" live in more than one space,
 * then group the domain elements into a single space, named kernelX, 
 * with X the kernel sequence numbers.
 *
 * We will first perform space-time transformation to transform the design to systolic array.
 * PE optimization is applied next including: array parititioning, latency hiding, and
 * SIMD vectorization.
 * For array partitioning, the marker "array" is added between the tile and point loops.
 * All the loops below the "array" marker will be mapped to FPGA device at once.
 * For latency hiding, SIMD vectorization, all the generated loops will be marked
 * "latency" and "SIMD".
 *
 * The linear branch between the kernel node and "array" mark may also have a "local" mark.
 * If present, the mapping to local memory is computed at this point. The "local" mark 
 * will be removed at the end of this function.
 *
 * Compute array reference groups for all arrays, set the local
 * array bounds based on the set of domain instances that reach 
 * the kernel node, check the total amount of shared memory used and compute 
 * all group tilings.
 *
 * We save a copy of the schedule that may influence the mappings to shared or private
 * memory in kernel->copy_schedule.
 *
 * We add copy statements to the schedule tree and create representations for the local
 * variables in the kernel.
 *
 * We keep a copy of the isl_id that points to the kernel to ensure 
 * that the kernel does not get destroyed if the schedule node 
 * is freed due to some error condition.
 */
static __isl_give isl_schedule_node *mark_kernels(
    struct polysa_gen *gen, __isl_take isl_schedule_node *node)
{
  isl_size num_sa = 0;
  struct polysa_kernel **sa_candidates;
  struct polysa_kernel *sa_opt, *kernel;
  isl_schedule *schedule;
  /* Enable for array partitioning, latency hiding, SIMD */
  bool pe_opt_en[3] = {1, 1, 1}; 
  isl_union_set *domain, *expanded;
  int single_statement;
  isl_union_map *host_schedule;
  isl_set *host_domain;
  isl_id *id;
  isl_union_pw_multi_aff *contraction;

  /* Generate systolic arrays using space-time mapping. */
  schedule = isl_schedule_node_get_schedule(node);
  sa_candidates = sa_space_time_transform(schedule, gen->prog->scop, &num_sa);
  if (num_sa > 0)
    printf("[PolySA] %d systolic arrays generated.\n", num_sa);

  /* Pick up one systolic array to proceed based on heuristics. */
  sa_opt = sa_candidates_smart_pick(sa_candidates, num_sa);

  /* Apply PE optimization. */
  pe_opt_en[1] = 0;
  pe_opt_en[2] = 0;
  sa_pe_optimize(sa_opt, pe_opt_en);
 
  node = isl_schedule_get_root(sa_opt->schedule);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_child(node, 0);
  kernel = sa_opt;

//  // debug
//  isl_printer *p = isl_printer_to_file(gen->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Insert "local" mark. */
  node = polysa_tree_insert_local_before_array(node);
  if (!node)
    return NULL;

  /* Create local arrays. */
  kernel = polysa_kernel_create_local_arrays(kernel, gen->prog);
  if (!kernel)
    return isl_schedule_node_free(node);

  domain = isl_schedule_node_get_domain(node);
  single_statement = isl_union_set_n_set(domain) == 1;

  /* Prepare some metadata. */
  kernel->prog = gen->prog;
  kernel->options = gen->options;
  kernel->context = extract_context(node, gen->prog);
  kernel->core = isl_union_set_universe(isl_union_set_copy(domain));
  contraction = isl_schedule_node_get_subtree_contraction(node);
  kernel->contraction = isl_union_pw_multi_aff_copy(contraction);
  expanded = isl_union_set_copy(domain);
  expanded = isl_union_set_preimage_union_pw_multi_aff(expanded, contraction);
  kernel->expanded_domain = isl_union_set_copy(expanded);
  kernel->arrays = accessed_by_domain(expanded, gen->prog);
  kernel->id = gen->kernel_id++;
  /* For FPGA, we set grid_size and block_size as 1, i.e. only one thread block 
   * and one thread inside the thread block. */
  kernel->n_grid = 1;
  kernel->block_dim[0] = 1;
  kernel->n_block = 1;
  kernel->grid_dim[0] = 1;
  kernel->grid_size = extract_grid_size(kernel, isl_union_set_copy(domain));

  // compute_sync_writes()

  host_schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
  host_domain = isl_set_from_union_set(isl_union_map_range(host_schedule));

  /* Make all the host loops atomic so that kernel is only called once. */
  node = atomic_ancestors(node);

  id = isl_id_alloc(gen->ctx, "kernel", kernel);
  id = isl_id_set_free_user(id, &polysa_kernel_free_wrap);
  node = isl_schedule_node_insert_mark(node, isl_id_copy(id));

  if (!single_statement)
    node = group_statements(node, kernel->id); 

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Current at the kernel mark, no need to move up */
  // node = polysa_tree_move_up_to_kernel(node);

  if (polysa_group_references(kernel, node) < 0) 
    node = isl_schedule_node_free(node);
  /* Localize the array bounds using parameters from the host domain */
  localize_bounds(kernel, host_domain); 
  isl_set_free(host_domain);

  check_local_memory_bound(kernel); 
  mark_global_arrays(kernel); 
  compute_group_tilings(kernel); 

  /* Save a copy of copy_schedule */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  kernel->copy_schedule_dim = isl_schedule_node_get_schedule_depth(node);
  kernel->copy_schedule = 
    isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
  contraction = isl_union_pw_multi_aff_copy(kernel->contraction);
  kernel->copy_schedule = 
    isl_union_pw_multi_aff_pullback_union_pw_multi_aff(
        kernel->copy_schedule, contraction);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  node = polysa_tree_move_up_to_kernel(node);
  node = add_copies(kernel, node); 

  /* Delete the local node */
  node = polysa_tree_move_down_to_local(node, kernel->core); 
  node = isl_schedule_node_delete(node);

  node = polysa_tree_move_up_to_kernel(node);

  if (create_kernel_vars(kernel) < 0) 
    node = isl_schedule_node_free(node);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  if (!single_statement)
    node = isl_schedule_node_parent(node);
//  node = isl_schedule_node_parent(node);

  isl_id_free(id);
  if (!id)
    polysa_kernel_free(kernel);

  return node;
}

/* Select the best "schedule" for mapping to FPGA.
 *
 * Unlike PPCG, in PolySA, only one SA kernel is created out of the 
 * original program, which is guaranteed by the previous step.
 * We will insert a context node, create a polysa_kernel for the schedule tree
 * beneath. Nodes for copying arrays in and out of the FPGA device and for
 * initializing and clearing the device are added. 
 *
 * The FPGA code is generated in a context where at least one statement 
 * instance is executed. The corresponding guard is inserted around 
 * the entire schedule.
 */
__isl_give isl_schedule *sa_map_to_device(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule)
{
  isl_schedule_node *node;
  isl_set *context;
  isl_set *guard;
  isl_union_set *domain;
  isl_union_map *prefix;
  isl_union_pw_multi_aff *contraction;
  struct polysa_prog *prog;

  context = isl_set_copy(gen->prog->context);
  context = isl_set_from_params(context);
  schedule = isl_schedule_insert_context(schedule, context);

  prog = gen->prog;
  guard = isl_union_set_params(isl_union_set_copy(prog->scop->domain));
  prog->context = isl_set_intersect(prog->context, isl_set_copy(guard));
  guard = isl_set_from_params(guard);

  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_child(node, 0);
  domain = isl_schedule_node_get_domain(node);
  contraction = isl_schedule_node_get_subtree_contraction(node);
  domain = isl_union_set_preimage_union_pw_multi_aff(domain,
            isl_union_pw_multi_aff_copy(contraction));
  prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
            contraction);
  
  node = mark_kernels(gen, node); 
  // debug
  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_node(p, node);
  printf("\n");
  // debug
  node = sa_add_to_from_device(node, domain, prefix, gen->prog);
  node = isl_schedule_node_root(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_guard(node, guard);
  node = isl_schedule_node_child(node, 0);

  node = sa_add_init_clear_device(node);

  // debug
  p = isl_printer_print_schedule_node(p, node);
  printf("\n");
  // debug
 
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  return schedule;
}

/* Generate HLS code for "scop" and print it to "p".
 * After generating an AST for the transformed scop as explained below,
 * we call "gen->print" to print the AST in the desired output format 
 * to "p".
 * 
 * If it turns out that it does not make sense to generate SA code, 
 * then we generate CPU code instead.
 * 
 * The declarations of the arrays that are visible outside of the scop
 * are printed outside of the code generated from the schedule,
 * because the generated code may involve a guard around the entire code.
 * 
 * We first compute a schedule that respects the dependences 
 * of the original program and test if the current program can be mapped to sa.
 * If not, we will generate CPU code instead.
 * If the --load-schedule is specified, then the loaded schedule 
 * is used instead of a computed schedule.
 * 
 * For the candidate program, a sequence of optimizations are performed, 
 * including: 
 * - Array Generation
 * - PE Optimization
 *   - Array Partitioning
 *   - Latency Hiding
 *   - SIMD Vectorization
 * - Data Transfer Optimization
 *   - I/O analysis
 *   - Interior I/O Elimination
 * 
 * After the array partitioning, we have program with
 * K
 * |
 * T
 * |
 * P
 * 
 * We add the kernel marker on top.
 * For each iteration of the T band and for each array, we compute
 * the array elements accessed by that iteration, construct a rectangular
 * box around it and shift it to the origin. The result is used
 * as the on-chip memory for the array.
 * 
 * Copying statements are added to this schedule tree.
 * In practice, these are added in front of the P band, but some of them 
 * my get hoisted up to higher levels.
 * 
 * The entire AST is then generated from the single resulting schedule tree.
 * During the generation the subtrees at kernel nodes (K) are saved aside and
 * replaced by kernel calls. The result is printed as host code while the saved
 * subtrees are printed as device code.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
  struct polysa_gen *gen, struct ppcg_scop *scop,
  struct ppcg_options *options)
{
  struct polysa_prog *prog;
  isl_ctx *ctx;
  isl_schedule *schedule;
  isl_bool any_sa;

  if (!scop) 
    return isl_printer_free(p);
  
  ctx = isl_printer_get_ctx(p);
  prog = polysa_prog_alloc(ctx, scop);
  if (!prog)
    return isl_printer_free(p);

  gen->prog = prog;
  schedule = get_schedule(gen); 

  isl_bool is_legal = sa_legality_check(schedule, scop);
  if (is_legal < 0 || !is_legal) {
    if (is_legal < 0)
      p = isl_printer_free(p);
    else 
      p = print_cpu(p, scop, options);
    isl_schedule_free(schedule);
  } else {
    /* Perform opt stages:
     * Array Generation -> PE Optimization 
     * -> Data Transfer Optimization -> Design Optimizer
     */    
    schedule = sa_map_to_device(gen, schedule);
    /* Generate the AST tree. */
    gen->tree = sa_generate_code(gen, schedule);

    // debug
    isl_printer *p_d = isl_printer_to_file(isl_ast_node_get_ctx(gen->tree), stdout);
    p_d = isl_printer_set_output_format(p_d, ISL_FORMAT_C);
    p_d = isl_printer_print_ast_node(p_d, gen->tree);
    printf("\n");
    // debug

    p = ppcg_set_macro_names(p);
    p = ppcg_print_exposed_declarations(p, prog->scop);
    p = gen->print(p, gen->prog, gen->tree, &gen->types, 
            gen->print_user);


    isl_ast_node_free(gen->tree);
  }

  // TODO
//  p = print_cpu(p, scop, options);
//  isl_schedule_free(schedule);

  polysa_prog_free(prog);
  
  return p;
}

/* Wrapper around generate for use as a ppcg_transform callback. 
 */
static __isl_give isl_printer *generate_wrap(__isl_take isl_printer *p,
  struct ppcg_scop *scop, void *user)
{
  struct polysa_gen *gen = user;

  return generate(p, gen, scop, gen->options);
}

/* Transform the code in the file called "input" by replacing 
 * all scops by corresponding FPGA code and write the results to "out".
 */
int generate_sa(isl_ctx *ctx, const char *input, FILE *out, 
  struct ppcg_options *options,
  __isl_give isl_printer *(*print)(__isl_take isl_printer *p,
    struct polysa_prog *prog, __isl_keep isl_ast_node *tree,
    struct polysa_types *types, void *user), void *user)
{
  struct polysa_gen gen;  
  int r;
  int i;

  gen.ctx = ctx;
  gen.sizes = extract_sizes_from_str(ctx, options->sizes);
  gen.options = options;
  gen.kernel_id = 0;
  gen.print = print;
  gen.print_user = user;
  gen.types.n = 0;
  gen.types.name = NULL;

  if (options->debug->dump_sizes) {
    isl_space *space = isl_space_params_alloc(ctx, 0);
    gen.used_sizes = isl_union_map_empty(space);
  }

  r = ppcg_transform(ctx, input, out, options, &generate_wrap, &gen);

  if (options->debug->dump_sizes) {
    isl_union_map_dump(gen.used_sizes);
    isl_union_map_free(gen.used_sizes);
  }

  isl_union_map_free(gen.sizes);
  for (i = 0; i < gen.types.n; ++i)
    free(gen.types.name[i]);
  free(gen.types.name);
  
  return r;
}
