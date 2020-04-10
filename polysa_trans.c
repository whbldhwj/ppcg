#include "polysa_trans.h"
#include "polysa_device.h"
#include "polysa_array_tile.h"

static __isl_give isl_schedule_node *unroll(__isl_take isl_schedule_node *node)
{
  int n;

  n = isl_schedule_node_band_n_member(node);
  for (int i = 1; i < n; ++i) {
    node = isl_schedule_node_band_member_set_ast_loop_type(node, i, isl_ast_loop_unroll);
  }

  return node;
}

static void free_group_pair(void *user)
{
  struct polysa_array_ref_group_pair *pair = user;
  free(pair);
}

static struct polysa_array_ref_group *polysa_find_pe_group(
  struct polysa_local_array_info *local_array,
  struct polysa_array_ref_group *io_group, 
  struct polysa_stmt_access *ref)
{
  if (local_array->array_type == POLYSA_INT_ARRAY)
    return local_array->pe_groups[0];
  
  for (int i = 0; i < local_array->n_pe_group; i++) {
    struct polysa_array_ref_group *pe_group = local_array->pe_groups[i];
    if (pe_group->refs[0] == ref)
      return pe_group;
  }

  return NULL;
}

/* Apply array partitioning.
 * Apply loop tiling on the band that contains the space loops
 * Reorganize the array partitioning loops and place them following the
 * ascending order of the dependence distances. 
 */
isl_stat sa_array_partitioning_optimize(struct polysa_kernel *sa, 
  bool en, char *mode, bool L2_en, char *L2_mode)
{
  int tile_len;
  isl_schedule *schedule;
  int *tile_size;
  isl_id *id;

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

  if (!en) {
    /* Add the array marker */
    id = isl_id_alloc(sa->ctx, "array", NULL);
    node = isl_schedule_node_insert_mark(node, id);
    
    isl_schedule_free(sa->schedule);
    sa->schedule = isl_schedule_node_get_schedule(node);
    isl_schedule_node_free(node);
    return isl_stat_ok;
  }

  printf("[PolySA] Apply array partitioning.\n");

//  // debug
//  isl_printer *pd = isl_printer_to_file(sa->ctx, stdout);
//  pd = isl_printer_set_yaml_style(pd, ISL_YAML_STYLE_BLOCK);
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
 
  tile_len = isl_schedule_node_band_n_member(node);
  if (!strcmp(mode, "manual")) {
    tile_size = read_array_part_tile_sizes(sa, tile_len) ;
    if (!tile_size) {
      /* Dump out the number and upper bounds of array_part loops and exit the program */
      int *ubs = extract_band_upper_bounds(sa, node);
      FILE *fp;
      char *content;
      cJSON *tuning, *array_part_json, *loops_json;

      tuning = cJSON_CreateObject();
      array_part_json = cJSON_CreateObject();      
      cJSON_AddItemToObject(tuning, "array_part", array_part_json);
      loops_json = cJSON_CreateArray();
      cJSON_AddItemToObject(array_part_json, "tilable_loops", loops_json);
      for (int i = 0; i < tile_len; i++) {
        cJSON *loop = cJSON_CreateNumber(ubs[i]);
        cJSON_AddItemToArray(loops_json, loop);
      }
      fp = fopen("polysa.tmp/tuning.json", "w");
      content = cJSON_Print(tuning);
      fprintf(fp, "%s", content);
      cJSON_Delete(tuning);
      exit(0);
    }   
  } else {
    /* Perform the array partitioning following the default policy */
    tile_size = read_default_array_part_tile_sizes(sa, tile_len);
  }

//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  /* Tile the band. */  
  if (!tile_size) {
    isl_schedule_node_free(node);
    return isl_stat_error;
  }
  
  if (sa->type == POLYSA_SA_TYPE_SYNC) {
    for (int i = 0; i < sa->n_sa_dim; i++) {
      sa->sa_dim[i] = tile_size[tile_len - sa->n_sa_dim + i];
    }
  } else {
    for (int i = 0; i < sa->n_sa_dim; i++) {
      sa->sa_dim[i] = tile_size[i];
    }
  }

  /* Examine if array size is 1 at any dimension */
  for (int i = 0; i < sa->n_sa_dim; i++) {
    if (sa->sa_dim[i] == 1) {
      /* Skip the array partition */
      id = isl_id_alloc(sa->ctx, "array", NULL);
      node = isl_schedule_node_insert_mark(node, id);

      free(tile_size);
      isl_schedule_free(sa->schedule);
      sa->schedule = isl_schedule_node_get_schedule(node);
      isl_schedule_node_free(node);
      return isl_stat_ok;
    }
  }

  node = polysa_tile_band(node, tile_size);
  free(tile_size);
    
//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  /* Add the array marker */
  node = isl_schedule_node_child(node, 0);
  id = isl_id_alloc(sa->ctx, "array", NULL);
  node = isl_schedule_node_insert_mark(node, id);
  node = isl_schedule_node_parent(node);

  /* Examine if there is any flow dep carried in the array_part band */
  if (!sa->options->credit_control) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      if (!isl_schedule_node_band_member_get_coincident(node, i)) {
        printf("[PolySA] WARNING: Flow deps carried in the array partition band.\n");
        printf("[PolySA] WARNING: Using simple task pipelining could lead to potential data hazards.\n");
        printf("[PolySA] WARNING: The program will proceed as usual. You could consider enabling credit control.\n");
        break;
      }
    }
  }
  if (sa->options->credit_control) {
    printf("[PolySA] ERROR: Credit control is not supported yet!\n");
    exit(1);
    // TODO: modify the schedule to add credit rd/wr for I/O modules
    // TODO: modify the module decls and fifo decls for credit fifos
//    /* Disable double-buffering */
//    sa->options->double_buffer = 0;
  }

  if (sa->options->two_level_buffer) {
    if (L2_en) {
      /* Tile the band again */
      printf("[PolySA] Two-level buffering is set. Apply second-level array partitioning.\n");
      tile_len = isl_schedule_node_band_n_member(node);
      if (!strcmp(mode, "manual")) {
        tile_size = read_array_part_L2_tile_sizes(sa, tile_len);
        if (!tile_size) {
          /* Dump out the number of and upper bounds of array_part loops and exit the program */
          int *ubs = extract_band_upper_bounds(sa, node);
          FILE *fp;
          char *content;
          cJSON *tuning, *array_part_json, *loops_json;
  
          tuning = cJSON_CreateObject();
          array_part_json = cJSON_CreateObject();
          cJSON_AddItemToObject(tuning, "array_part_L2", array_part_json);
          loops_json = cJSON_CreateArray();
          cJSON_AddItemToObject(array_part_json, "tilable_loops", loops_json);
          for (int i = 0; i < tile_len; i++) {
            cJSON *loop = cJSON_CreateNumber(ubs[i]);
            cJSON_AddItemToArray(loops_json, loop);
          }
          fp = fopen("polysa.tmp/tuning.json", "w");
          content = cJSON_Print(tuning);
          fprintf(fp, "%s", content);
          cJSON_Delete(tuning);
          exit(0);
        }
      } else {
        /* Perform second-level array partitioning following the default policy */
        tile_size = read_default_array_part_L2_tile_sizes(sa, tile_len);
      }
  
      if (!tile_size) {
        isl_schedule_node_free(node);
        return isl_stat_error;
      }
      node = polysa_tile_band(node, tile_size);
      free(tile_size);
  
      /* Add the second-level array marker */
      node = isl_schedule_node_child(node, 0);
      id = isl_id_alloc(sa->ctx, "array_L2", NULL);
      node = isl_schedule_node_insert_mark(node, id);
      node = isl_schedule_node_parent(node);
    } else {
      /* Disable the L2 array partitioning */
      sa->options->two_level_buffer = 0;
    }
  }

//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  /* Clean up the band pe_opt properties. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &clear_pe_opt_prop, NULL);

  isl_schedule_free(sa->schedule);
  sa->schedule = schedule;

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

struct count_latency_hiding_loop_data {
  int tile_len;
  int *ubs;
  struct polysa_kernel *kernel;
};

static isl_bool count_latency_hiding_loop(__isl_keep isl_schedule_node *node, void *user)
{
  struct count_latency_hiding_loop_data *data = user;
  isl_schedule_node *node_copy;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    int n = isl_schedule_node_band_n_member(node);
    for (int i = 0; i < n; i++) {
      if (isl_schedule_node_band_member_get_pe_opt(node, i) == polysa_loop_latency) {
        data->tile_len = data->tile_len + 1; 
//        *cnt = *cnt + 1;
        /* Extract the loop upper bound */
        node_copy = isl_schedule_node_copy(node);
        if (i > 0) {
          node_copy = isl_schedule_node_band_split(node_copy, i);
          node_copy = isl_schedule_node_child(node_copy, 0);          
        }
        if (n - i - 1 > 0) {
          node_copy = isl_schedule_node_band_split(node_copy, 1);
        }
        int *ubs = extract_band_upper_bounds(data->kernel, node_copy);
        data->ubs = (int *)realloc(data->ubs, sizeof(int) * data->tile_len);
        data->ubs[data->tile_len - 1] = ubs[0];
        isl_schedule_node_free(node_copy);
        free(ubs);
      }
    }
  } 
  
  return isl_bool_true;
}

struct stride_coalesced_data {
  struct polysa_kernel *kernel;
  isl_union_map *prefix;
  int score;
};

static __isl_give isl_map *same(__isl_take isl_space *domain_space, int pos)
{
  isl_space *space;
  isl_aff *aff;
  isl_multi_aff *next;

  space = isl_space_map_from_set(domain_space);
  next = isl_multi_aff_identity(space);
  
  return isl_map_from_multi_aff(next);
}

/* Construct a map from domain_space to domain_space that increments
 * the dimension at position "pos" and leaves all other dimensions constant. 
 */
static __isl_give isl_map *next(__isl_take isl_space *domain_space, int pos)
{
  isl_space *space;
  isl_aff *aff;
  isl_multi_aff *next;

  space = isl_space_map_from_set(domain_space);
  next = isl_multi_aff_identity(space);
  aff = isl_multi_aff_get_aff(next, pos);
  aff = isl_aff_add_constant_si(aff, 1);
  next = isl_multi_aff_set_aff(next, pos, aff);

  return isl_map_from_multi_aff(next);
}

static int access_is_stride_zero(__isl_keep isl_map *access, int pos)
{
  isl_space *space;
  int dim;
  isl_map *next_element, *map, *next_iter;
  isl_set *accessed;
  int empty, zero;

  space = isl_map_get_space(access);
  space = isl_space_range(space);
  dim = isl_space_dim(space, isl_dim_set);
  if (dim == 0)
    next_element = isl_map_empty(isl_space_map_from_set(space));
  else
    next_element = same(space, pos);

  accessed = isl_map_range(isl_map_copy(access));
  map = isl_map_copy(next_element);
  map = isl_map_intersect_domain(map, isl_set_copy(accessed));
  map = isl_map_intersect_range(map, accessed);
  empty = isl_map_is_empty(map);
  isl_map_free(map);

  if (empty < 0 || empty) {
    isl_map_free(next_element);
    return empty;
  } 

  space = isl_map_get_space(access);
  space = isl_space_domain(space);
  next_iter = next(space, isl_map_dim(access, isl_dim_in) - 1);
  map = isl_map_apply_domain(next_iter, isl_map_copy(access));
  map = isl_map_apply_range(map, isl_map_copy(access));
  zero = isl_map_is_subset(map, next_element);

  isl_map_free(next_element);
  isl_map_free(map);

  return zero;
}

static int access_is_stride_one(__isl_keep isl_map *access, int pos)
{
  isl_space *space;
  int dim;
  isl_map *next_element, *map, *next_iter;
  isl_set *accessed;
  int empty, coalesced;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_map_get_ctx(access), stdout);
//  // debug

  space = isl_map_get_space(access);
  space = isl_space_range(space);
  dim = isl_space_dim(space, isl_dim_set);
  if (dim == 0)
    next_element = isl_map_empty(isl_space_map_from_set(space));
  else
    next_element = next(space, pos);

//  // debug
//  p = isl_printer_print_map(p, access);
//  printf("\n");
//  p = isl_printer_print_map(p, next_element);
//  printf("\n");
//  // debug

  accessed = isl_map_range(isl_map_copy(access));
  map = isl_map_copy(next_element);
  map = isl_map_intersect_domain(map, isl_set_copy(accessed));
  map = isl_map_intersect_range(map, accessed);
  empty = isl_map_is_empty(map);
  isl_map_free(map);

  if (empty < 0 || empty) {
    isl_map_free(next_element);
    return empty;
  } 

  space = isl_map_get_space(access);
  space = isl_space_domain(space);
  next_iter = next(space, isl_map_dim(access, isl_dim_in) - 1);
//  // debug
//  p = isl_printer_print_map(p, next_iter);
//  printf("\n");
//  // debug
  map = isl_map_apply_domain(next_iter, isl_map_copy(access));
  map = isl_map_apply_range(map, isl_map_copy(access));
  coalesced = isl_map_is_subset(map, next_element);

  isl_map_free(next_element);
  isl_map_free(map);

  return coalesced;
}

static isl_bool is_stride_coalesced_stmt(__isl_keep isl_set *set, void *user)
{
  isl_space *space;
  isl_id *id;
  struct polysa_stmt *stmt;
  struct stride_coalesced_data *data = user;
  struct polysa_stmt_access *accesses, *access;
  isl_map *prefix;

//  // debug
//  isl_printer *p = isl_printer_to_file(data->kernel->ctx, stdout);
//  // debug

  space = isl_set_get_space(set);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  isl_space_free(space);

//  // debug
//  p = isl_printer_print_id(p, id);
//  printf("\n");
//  // debug

  prefix = isl_map_from_union_map(isl_union_map_intersect_domain(
        isl_union_map_copy(data->prefix), isl_union_set_from_set(isl_set_copy(set))));

  stmt = find_stmt(data->kernel->prog, id);
  isl_id_free(id);
  accesses = stmt->accesses;
  for (access = accesses; access; access = access->next) {
    isl_map *acc;
    int n;
    isl_bool is_zero = isl_bool_false, is_one = isl_bool_false;
    isl_pw_multi_aff *pma;
    int i;

    /* Skip the scalar access */
    if (access->n_index == 0)
      continue;

    /* Transform the access function */
    acc = isl_map_copy(access->access);
    acc = isl_map_apply_domain(acc, isl_map_copy(prefix));

    for (i = access->n_index - 1; i >= 0; i--) {
      is_zero = access_is_stride_zero(acc, i);
      if (is_zero)
        break;
    }
    if (!is_zero) {
      for (i = access->n_index - 1; i >= 0; i--) {
        is_one = access_is_stride_one(acc, i);
        if (is_one)
          break;
      }
    }
   
    isl_map_free(acc);

    if (!(is_zero || is_one)) {
      isl_map_free(prefix);
      return isl_bool_false;
    } else {
      if (i == access->n_index - 1) {
        access->layout_trans = 0;
        access->simd_dim = i;
      } else {
        access->layout_trans = 1;
        access->simd_dim = i;
      }
      data->score = data->score + (1 - access->layout_trans);
//      access = is_zero? 0 : (is_one? 1: -1);
    }
  }
  
  isl_map_free(prefix);
  return isl_bool_true;
}

/* This function examines if the access function of the statements under the current "node"
 * has only stride-0/1 access.
 */
static isl_bool is_stride_coalesced_at_node(__isl_keep isl_schedule_node *node,
  void *user)
{
  struct stride_coalesced_data *data = user;
  struct polysa_kernel *kernel = data->kernel;
  isl_union_set *domain;
  isl_union_map *prefix;
  isl_bool one_or_zero;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return isl_bool_true;

  domain = isl_schedule_node_get_domain(node);
  prefix = isl_schedule_node_get_prefix_schedule_union_map(node); 
  data->prefix = prefix;

  /* Examine each statment under the loop */
  one_or_zero = isl_union_set_every_set(domain, &is_stride_coalesced_stmt, data);

  isl_union_map_free(data->prefix);
  isl_union_set_free(domain);

  return one_or_zero;
}

/* This function calculates the score of the current "node" in terms of 
 * opportunities of SIMD vectorization.
 * First of all, the loop has either to be a parallel loop or a reduction loop.
 * We test the reduction loop by examining if the carried dependence by the 
 * current loop is from a reduction statement. 
 * We rely on users to mark the reduction statements in the reduction.annotation file.
 * Next, we will test if all the array references under the current loop
 * has only stride-0/1 access.
 * If either of the two criteria fails, the loop is non-vectorizable.
 * Finally, we calculate the score of the current loop which is used to rank
 * the loop given there are multiple SIMD loops available.
 * The scores is calculated as:
 * score = Sigma_{all_array_references_under_the_loop} 
 *           (is_access_stride-0/1 * (1 - is_layout_transformation_required)
 *              + 2 * is_loop_parallel + 4 * is_loop_reduction) 
 * With the current heuristic, we favor reduction loop over parallel loop,
 * because the reduction loop introduces less overhead than the parallel loop.
 * And we favor loops that doesn't require layout transformation.
 */
static int is_stride_coalesced(__isl_keep isl_schedule_node *node, struct polysa_kernel *kernel, int *layout_transform)
{
  /* At each leaf node, examine if the access function of the statement has 
   * stride-0/1 access.
   */
  int score = 0;
  struct stride_coalesced_data data;
  isl_bool coalesced;

  data.kernel = kernel;
  data.score = score;
  coalesced = isl_schedule_node_every_descendant(node, &is_stride_coalesced_at_node, &data);
  
  /* Examine and make sure all the array references of the same array have the same
   * dimenison for layout transformation */
  if (coalesced) {
    struct polysa_kernel *kernel = data.kernel;
    for (int i = 0; i < kernel->n_array; i++) {
      struct polysa_local_array_info *local_array;
      int simd_dim = -1;
      local_array = &kernel->array[i];
      for (int j = 0; j < local_array->array->n_ref; j++) {
        struct polysa_stmt_access *acc = local_array->array->refs[j];
        if (acc->layout_trans == 1) {
          if (simd_dim == -1)
            simd_dim = acc->simd_dim;
          else {
            if (simd_dim != acc->simd_dim) {
              coalesced = 0;
              return coalesced? data.score : -1;
            }
          }
        }
      }
    }
  }

  /* Print out the layout transform information */
  if (coalesced) {
    struct polysa_kernel *kernel = data.kernel;
    isl_printer *p;

    p = isl_printer_to_file(kernel->ctx, stdout);
    for (int i = 0; i < kernel->n_array; i++) {
      struct polysa_local_array_info *local_array;
      local_array = &kernel->array[i];
      for (int j = 0; j < local_array->array->n_ref; j++) {
        struct polysa_stmt_access *acc = local_array->array->refs[j];
        
        if (acc->layout_trans != -1)  {
          printf("[PolySA] Array reference ");
          if (acc->read)
            printf("(R): ");
          else
            printf("(W): ");
          p = isl_printer_print_map(p, acc->access);
          printf("\n");
          if (acc->layout_trans == 1){
            printf("[PolySA] Layout transform at dim: %d\n", acc->simd_dim);
            *layout_transform = 1;
          }
          acc->layout_trans = -1;
          acc->simd_dim = -1;
        }
      }
    }
    isl_printer_free(p);
  }

  return coalesced? data.score : -1;
}

struct simd_vectorization_data {
  struct polysa_kernel *kernel;
  int *scores;
  int best_score;
  int layout_trans;
  int n_loops;
  int loop_cnt;
  char *mode;
  int *ubs;
  int *tile_size;
};

/* A loop is identified to be vectorizable if it is a parallel or 
 * reduction loop; with stride-0/1 access.
 * Only time loops are examined.
 */ 
static isl_schedule_node *detect_simd_vectorization_loop(__isl_take isl_schedule_node *node, void *user)
{
  struct simd_vectorization_data *data = user;
  struct polysa_kernel *sa = data->kernel;
  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  int score;
  isl_schedule_node *cur_node;
  int is_latency;

//  // debug
//  isl_printer *pd = isl_printer_to_file(ctx, stdout);
//  pd = isl_printer_set_yaml_style(pd, ISL_YAML_STYLE_BLOCK);
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  /* If the currrent node is under the latency mark, return
   * as we don't use latency hiding loop as candidates. */
  is_latency = is_node_under_latency(node);
  if (is_latency) 
    return node;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      if ((isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_time)) {
        /* Two types of loops that we are interested in.
         * Parallel loop.
         * Reduction loop in the innermost loop band */
        int is_parallel = 0;
        int is_reduction = 0;
        int layout_transform = 0;
        int score_i;

        if (!isl_schedule_node_band_member_get_coincident(node, i)) {
          /* Examine if the current node is the innermost node */
          node = isl_schedule_node_child(node, 0);
          isl_bool no_inner_band = isl_schedule_node_every_descendant(node, 
              &no_permutable_node, NULL);
          node = isl_schedule_node_parent(node);
          if (!no_inner_band) {
            /* Examine if all the loops inside are parallel loops */
            node = isl_schedule_node_child(node, 0);
            isl_bool all_parallel = isl_schedule_node_every_descendant(node,
                  &all_parallel_node, NULL);
            node = isl_schedule_node_parent(node);
            if (all_parallel) 
              no_inner_band = isl_bool_true;
          }

          if (no_inner_band && !strcmp(data->mode, "manual")) {            
            /* At present, we cannot analyze reduction loop by the PolySA.
             * We will print each node and take the user guidance.
             * Besides, we only check reduction loop in maunal mode.
             * In the auto mode, only parallel loops are examined.
             */
            isl_printer *p;
            char c;
             
            p = isl_printer_to_file(ctx, stdout);
            p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
            p = isl_printer_print_schedule_node(p, node);

            printf("[PolySA] Detecting the reduction loop.\n");
            printf("[PolySA] Band member position: %d\n", i);
            printf("[PolySA] Please input if the current loop is a reduction loop [y/n]: ");
            c = getchar();
            is_reduction = (c == 'y')? 1 : 0;
            isl_printer_free(p);
          } else {
            continue;
          }
        } else {
          is_parallel = 1;
        }

        /* Test if all the array references under the current loop 
         * has only stride-0/1 access. */
        if (is_parallel || is_reduction) {
          cur_node = node;
          node = isl_schedule_node_dup(cur_node);
          
          if (i > 0) {
            node = isl_schedule_node_band_split(node, i);
            node = isl_schedule_node_child(node, 0);
          }
          if (isl_schedule_node_band_n_member(node) - i - 1 > 0) {
            node = isl_schedule_node_band_split(node, 1);
          }

          /* Sink the band innermost */
          node = isl_schedule_node_band_sink(node);
//          // debug
//          pd = isl_printer_print_schedule_node(pd, node);
//          pd = isl_printer_flush(pd);
//          pd = isl_printer_print_schedule_node(pd, cur_node);
//          pd = isl_printer_flush(pd);
//          // debug
          score = 2 * is_parallel + 4 * is_reduction;
          score_i = is_stride_coalesced(node, sa, &layout_transform);          
          isl_schedule_node_free(node);
          node = cur_node;
          if (score_i < 0) {
            /* The array references are not coalesced */
            score = -1;
            continue;
          } else {
            score += score_i;
            printf("[PolySA] The loop is legal to be vectorized with score: %d\n", score);
            if (layout_transform) 
              printf("[PolySA] Layout transformation is required to proceed.\n");
            node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_simd); 

            if (score >= data->best_score) {
              data->best_score = score;
              data->layout_trans = layout_transform;
            }
            data->n_loops = data->n_loops + 1;
            data->scores = (int *)realloc(data->scores, sizeof(int) * data->n_loops);
            data->scores[data->n_loops - 1] = score;
            /* Extract the loop upper bounds */
//            // debug
//            pd = isl_printer_print_schedule_node(pd, node);
//            pd = isl_printer_flush(pd);
//            // debug
            int *ubs = extract_band_upper_bounds(sa, node);
            data->ubs = (int *)realloc(data->ubs, sizeof(int) * data->n_loops);
            data->ubs[data->n_loops - 1] = ubs[i];
            free(ubs);
          }
        }
      }
    }
  }

  return node;
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

/* Examine if the node is the last band node, if so, add a "simd" mark before the node. */
static __isl_give isl_schedule_node *add_simd_mark(__isl_take isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    node = isl_schedule_node_child(node, 0);
    isl_bool no_inner_band = isl_schedule_node_every_descendant(node, 
        &no_permutable_node, NULL);
    node = isl_schedule_node_parent(node);
    if (no_inner_band) {
      /* Insert the "simd" mark. */
      isl_id *id = isl_id_alloc(isl_schedule_node_get_ctx(node), "simd", NULL);
      node = isl_schedule_node_insert_mark(node, id);
    }
  }

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

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  int n;
  isl_id *id;
  n = isl_schedule_node_band_n_member(node);

  for (int i = n - 1; i >= 0; i--) {
    if (isl_schedule_node_band_member_get_pe_opt(node, i) == polysa_loop_latency) {      
      int loop_tile_size = data->tile_size[data->tile_len - data->n_touched_loop - 1];
      (data->n_touched_loop)++;
      /* Skip loop tile size as 1 */
      if (loop_tile_size > 1) {
        /* Tile the current loop and permute it to be the innermost time loop. */
        /* Tile the loop in the band at "i"th position with the size "loop_tile_size".
         * The returned node points at the tile loop. */
        node = polysa_node_band_tile_loop(node, loop_tile_size, i); 
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
//        p = isl_printer_flush(p);
//        // debug

        (data->n_tiled_loop)++;
        return node;
      } else {
        /* Reset the pe_opt property */
        node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
      }
    }
  }

  return node;
}

/* Insert "hls_pipeline" mark under the last time loop */
static __isl_give isl_schedule_node *add_hls_pipeline(
  __isl_take isl_schedule_node *node, void *user)
{
  struct polysa_kernel *sa = user;
  isl_ctx *ctx = sa->ctx;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_band) 
    return node;

  /* Examine if the node is innermost */
  node = isl_schedule_node_child(node, 0);
  isl_bool no_inner_band = isl_schedule_node_every_descendant(node, 
      &no_permutable_node, NULL);
  node = isl_schedule_node_parent(node);
  if (!no_inner_band)
    return node;

  int n = isl_schedule_node_band_n_member(node);

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  if (sa->type == POLYSA_SA_TYPE_ASYNC) {
    if (isl_schedule_node_band_member_get_space_time(node, n - 1) == polysa_loop_time) {
      isl_id *id;
//      // debug
//      p = isl_printer_print_schedule_node(p, node);
//      p = isl_printer_flush(p);
//      // debug
      id = isl_id_alloc(ctx, "hls_pipeline", NULL);
      node = isl_schedule_node_child(node, 0);
      node = isl_schedule_node_insert_mark(node, id);
      node = isl_schedule_node_parent(node);
//      // debug
//      p = isl_printer_print_schedule_node(p, node);
//      p = isl_printer_flush(p);
//      // debug
    }
  } else if (sa->type == POLYSA_SA_TYPE_SYNC) {
    if (isl_schedule_node_band_member_get_space_time(node, 0) != polysa_loop_time) {
      node = isl_schedule_node_parent(node);
      while (isl_schedule_node_get_type(node) != isl_schedule_node_band &&
          isl_schedule_node_has_parent(node)) {
        node = isl_schedule_node_parent(node);
      }
    }
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      n = isl_schedule_node_band_n_member(node);
      for (int i = n - 1; i >= 0; i--) {
        if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_time) {
          isl_id *id = isl_id_alloc(ctx, "hls_pipeline", NULL);
          if (i != n - 1) {
            node = isl_schedule_node_band_split(node, i + 1);            
          }
          node = isl_schedule_node_child(node, 0);
          node = isl_schedule_node_insert_mark(node, id);
          node = isl_schedule_node_parent(node);
          break;
        }
      }
    }
  }

  return node;
}

static __isl_give isl_schedule_node *polysa_latency_tile_loop(__isl_take isl_schedule_node *node, struct polysa_kernel *sa, char *mode)
{
  int tile_len;
  int *tile_size;
  struct count_latency_hiding_loop_data data;
  data.tile_len = 0;
  data.ubs = NULL;
  data.kernel = sa;
  int i;
  
  /* Count the candidate loop number. */
  isl_schedule_node_foreach_descendant_top_down(
      node, &count_latency_hiding_loop, &data);
  // printf("%d\n", tile_len);
  tile_len = data.tile_len;

  if (!strcmp(mode, "manual")) {
    tile_size = read_latency_tile_sizes(sa, tile_len);
    if (!tile_size) {
      /* Dump out the number and upper bounds of latency loops and exit the program */
      int *ubs = data.ubs;
      FILE *fp;
      char *content;
      cJSON *tuning, *latency_json, *loops_json;

      tuning = cJSON_CreateObject();
      latency_json = cJSON_CreateObject();
      cJSON_AddItemToObject(tuning, "latency", latency_json);
      loops_json = cJSON_CreateArray();
      cJSON_AddItemToObject(latency_json, "tilable_loops", loops_json);
      for (int i = 0; i < tile_len; i++) {
        cJSON *loop = cJSON_CreateNumber(ubs[i]);
        cJSON_AddItemToArray(loops_json, loop);
      }
      fp = fopen("polysa.tmp/tuning.json", "w");
      content = cJSON_Print(tuning);
      fprintf(fp, "%s", content);
      cJSON_Delete(tuning);
      exit(0);
    }
  } else {
    /* Perform the latency hiding following the default policy */
    tile_size = read_default_latency_tile_sizes(sa, tile_len);
  }

  free(data.ubs);
  if (!tile_size) {
    isl_schedule_node_free(node);
    return NULL;
  }

  /* Examine if all the tiling factors are 1, in that case, we will
   * skip the tiling and split off the last time dimension to add a 
   * hls_pipeline mark. */
  for (i = 0; i < tile_len; i++) {
    if (tile_size[i] != 1)
      break;
  }
  if (i == tile_len) {
    node = isl_schedule_node_map_descendant_bottom_up(node, 
        &add_hls_pipeline, sa);
  } else {
    /* Tile the loop. */
    struct polysa_pe_opt_tile_data tile_data = {0, 0, tile_len, tile_size, sa};
    while (tile_data.n_touched_loop != tile_len) {
      node = isl_schedule_node_map_descendant_bottom_up(
        node, &polysa_latency_tile_band_loop, &tile_data);
    }
  }

  free(tile_size);
  return node;
}

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

struct latency_opt_check_data {
  struct polysa_kernel *kernel;
  int is_required;
};

/* Check if the innermost time loop is parallel or not.
 * If this loop is parallel, it can be used for latency hiding and 
 * there is no need for further optimization.
 * We will split off this loop from the band, and attach a "latency"
 * marker above it.
 */
static __isl_give isl_schedule_node *latency_opt_check(
  __isl_take isl_schedule_node *node, void *user)
{
  struct latency_opt_check_data *data = user;
  struct polysa_kernel *sa = data->kernel;
  isl_ctx *ctx = sa->ctx;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return node;

  /* Examine if the node is innermost */
  node = isl_schedule_node_child(node, 0);
  isl_bool no_inner_band = isl_schedule_node_every_descendant(node,
      &no_permutable_node, NULL);
  node = isl_schedule_node_parent(node);
  if (!no_inner_band) 
    return node;

  int n = isl_schedule_node_band_n_member(node);

  if (sa->type == POLYSA_SA_TYPE_ASYNC) {
    if (isl_schedule_node_band_member_get_coincident(node, n - 1) &&
        isl_schedule_node_band_member_get_space_time(node, n - 1) == polysa_loop_time) {
      isl_id *id;
      data->is_required = 0;
      /* Split off the loop and attach a "latency" mark */
      if (n > 1) {
        node = isl_schedule_node_band_split(node, n - 1);
        node = isl_schedule_node_child(node, 0);
      }
      id = isl_id_alloc(ctx, "latency", NULL);
      node = isl_schedule_node_insert_mark(node, id);
      node = isl_schedule_node_parent(node);
    }
  } else if (sa->type == POLYSA_SA_TYPE_SYNC) {
    if (isl_schedule_node_band_member_get_space_time(node, 0) != polysa_loop_time) {
      node = isl_schedule_node_parent(node);
      while (isl_schedule_node_get_type(node) != isl_schedule_node_band &&
          isl_schedule_node_has_parent(node)) {
        node = isl_schedule_node_parent(node);
      }      
    }
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      n = isl_schedule_node_band_n_member(node);
      for (int i = n - 1; i >= 0; i--) {
        if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_time) {
          if (isl_schedule_node_band_member_get_coincident(node, i)) {
            isl_id *id;
            data->is_required = 0;
            /* Split off the time loop */
            if (i > 1) {
              node = isl_schedule_node_band_split(node, i);
              node = isl_schedule_node_child(node, 0);
            }
            if (n - i - 1 > 0) {
              node = isl_schedule_node_band_split(node, 1);
            }
            id = isl_id_alloc(ctx, "latency", NULL);
            node = isl_schedule_node_insert_mark(node, id);
            node = isl_schedule_node_parent(node);
          }
          break;
        }
      }
    }
  }

  return node;
}

static isl_bool find_latency_mark(__isl_keep isl_schedule_node *node, void *user)
{
  if (isl_schedule_node_get_type(node) == isl_schedule_node_mark) {
    isl_id *id;

    id = isl_schedule_node_mark_get_id(node);
    if (!strcmp(isl_id_get_name(id), "latency")) {
      isl_id_free(id);
      return isl_bool_false;
    }
    isl_id_free(id);
  }

  return isl_bool_true;
}

/* Insert a "hls_pipeline" mark after the innermost "latency" mark.
 * The loop will be eventually pipelined.
 * The "hls_pipeline" mark is placed under the band node.
 */
static __isl_give isl_schedule_node *insert_pipeline_mark(
  __isl_take isl_schedule_node *node, void *user)
{
  struct polysa_kernel *kernel = user;
  isl_ctx *ctx = kernel->ctx;
  
  if (isl_schedule_node_get_type(node) == isl_schedule_node_mark) {
    isl_id *id;

    id = isl_schedule_node_mark_get_id(node);
    if (!strcmp(isl_id_get_name(id), "latency")) {
      /* Examine if there is any latency mark inside the current mark */
      isl_bool no_inner_latency;
      node = isl_schedule_node_child(node, 0);
      no_inner_latency = isl_schedule_node_every_descendant(node,
          &find_latency_mark, NULL); 
      node = isl_schedule_node_parent(node);
      if (no_inner_latency) {
        /* Insert the "hls_pipeline" mark below the band node */
        isl_id *hls_id;
        hls_id = isl_id_alloc(ctx, "hls_pipeline", NULL);
        node = isl_schedule_node_child(node, 0);
        node = isl_schedule_node_child(node, 0);
        node = isl_schedule_node_insert_mark(node, hls_id);

        node = isl_schedule_node_parent(node);
        node = isl_schedule_node_parent(node);
      }
    }
    isl_id_free(id);
  }

  return node;
}

/* Insert a "hls_unroll" mark after the "simd" mark.
 * The loop will be eventually unrolled.
 */
static __isl_give isl_schedule_node *insert_unroll_mark(
  __isl_take isl_schedule_node *node, void *user)
{
  struct polysa_kernel *kernel = user;
  isl_ctx *ctx = kernel->ctx;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_mark) {
    isl_id *id;

    id = isl_schedule_node_mark_get_id(node);
    if (!strcmp(isl_id_get_name(id), "simd")) {
      isl_id *hls_id;
      hls_id = isl_id_alloc(ctx, "hls_unroll", NULL);
      node = isl_schedule_node_child(node, 0);
      node = isl_schedule_node_child(node, 0);
      node = isl_schedule_node_insert_mark(node, hls_id);
      node = isl_schedule_node_parent(node);
      node = isl_schedule_node_parent(node);
    }
    isl_id_free(id);
  }

  return node;
}

/* Apply latency hiding. 
 * Go through all the loops, if there is any parallel loop (considering only RAW), 
 * such a loop will be identified as latency hiding loop candidate. Such loops will be
 * tiled. The point loops will be permuted as the innermost time loops.
 */
isl_stat sa_latency_hiding_optimize(struct polysa_kernel *sa, bool en, char *mode)
{
  isl_bool opt_required;
  isl_schedule *schedule = sa->schedule;
  isl_schedule_node *node = isl_schedule_get_root(schedule);

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  if (!en) {
    /* Split off the last time loop and add a hls_pipeline mark */
    node = isl_schedule_node_map_descendant_bottom_up(node,
        &add_hls_pipeline, sa);

    isl_schedule_free(sa->schedule);
    sa->schedule = isl_schedule_node_get_schedule(node);
    isl_schedule_node_free(node);
    return isl_stat_ok;
  }

  printf("[PolySA] Apply latency hiding.\n");
  
  /* Move down to the array marker. */
  node = polysa_tree_move_down_to_array(node, sa->core);
 
  /* Check if the innermost time loop is parallel loop.
   * If so, there is no need to perform latency hiding, safely reutrn.
   */
  struct latency_opt_check_data data;
  data.kernel = sa;
  data.is_required = 1;
  node = isl_schedule_node_map_descendant_bottom_up(node, 
      &latency_opt_check, &data);

  if (!data.is_required) {
    isl_schedule_free(schedule);
    schedule = isl_schedule_node_get_schedule(node);
    isl_schedule_node_free(node);
    sa->schedule = schedule;
    return isl_stat_ok;
  }

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
  node = polysa_latency_tile_loop(node, sa, mode);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Clean up the band pe_opt properties. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &clear_pe_opt_prop, NULL);
  
  sa->schedule = schedule;

  return isl_stat_ok;
}

struct data_transfer_opt_data {
  struct polysa_stmt_access *access;
  struct polysa_kernel *kernel;
  enum polysa_dep_type dep_type;
  isl_bool is_update;
};

struct dep_space_test_internal_data {
  isl_vec *dirvec;
  isl_basic_map *dep;
};

/* This function tests if the current node contains any space loop.
 * If so, test if the dependence is carried by the space loops, and update the 
 * dependence distance vector. 
 * If the dependence is carried at the space loop, return false,
 * else return true.
 */
isl_bool not_carried_at_space(__isl_keep isl_schedule_node *node, void *user)
{
  struct dep_space_test_internal_data *data = (struct dep_space_test_internal_data *)user;
  isl_basic_map *dep = data->dep;
  isl_basic_map *untagged_dep = isl_basic_map_from_map(isl_map_factor_domain(isl_map_from_basic_map(isl_basic_map_copy(dep))));
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band) {
    isl_basic_map_free(untagged_dep);
    return isl_bool_true;
  }

  int n_dim = isl_schedule_node_band_n_member(node);
  int n_space_dim, space_dim_start;
  n_space_dim = 0;
  for (int i = 0; i < n_dim; i++) {
    if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_space) {
      if (n_space_dim == 0)
        space_dim_start = i;
      n_space_dim++;
    }
  }

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  p = isl_printer_print_basic_map(p, untagged_dep);
//  printf("\n");
//  // debug

  if (n_space_dim > 0) {
    isl_vec *disvec = get_dep_dis_at_node(untagged_dep, node);
//    // debug
//    p = isl_printer_print_vec(p, disvec);
//    printf("\n");
//    // debug
    isl_vec *dirvec = isl_vec_zero(isl_schedule_node_get_ctx(node), n_space_dim);
    int carried = 0;
    for (int i = 0; i < n_space_dim; i++) {
      isl_val *val = isl_vec_get_element_val(disvec, space_dim_start + i);      
      dirvec = isl_vec_set_element_si(dirvec, i, isl_val_get_num_si(val));
      if (isl_val_get_num_si(val) > 0)
        carried = 1;
      isl_val_free(val);
    }
    data->dirvec = dirvec;
    isl_vec_free(disvec);
    isl_basic_map_free(untagged_dep);
    if (carried)
      return isl_bool_false;
    else
      return isl_bool_true;
  }
  isl_basic_map_free(untagged_dep);
  return isl_bool_true;
}

/* If dependence is carried by the space loop, then mark it with the access as exterior I/O, 
 * update the dirvec.
 * Otherwise, mark it as the interior I/O, assign a dirvec to transfer teh data.
 */
isl_stat data_transfer_update(__isl_keep isl_basic_map *dep, struct data_transfer_opt_data *data) 
{
  struct polysa_stmt_access *access = data->access;
  struct polysa_kernel *kernel = data->kernel;
  isl_id *src_id, *dest_id;
  isl_space *space;
  isl_space *src_space, *dest_space;
  isl_schedule_node *node;

  /* Test if the access is associated with the current dep. */
  space = isl_basic_map_get_space(dep);
  src_space = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
  dest_space = isl_space_unwrap(isl_space_range(space));
  src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
  dest_id = isl_space_get_tuple_id(dest_space, isl_dim_out);
  isl_space_free(src_space);
  isl_space_free(dest_space);

  if (src_id != access->ref_id && dest_id != access->ref_id) {
    isl_id_free(src_id);
    isl_id_free(dest_id);
    return isl_stat_ok;
  }
  isl_id_free(src_id);
  isl_id_free(dest_id);

  /* Test if the dependence is carried at the space loop. */
  struct dep_space_test_internal_data internal_data = { NULL, dep };
  node = isl_schedule_get_root(kernel->schedule);
  isl_bool is_carried_at_space = !isl_schedule_node_every_descendant(node, not_carried_at_space, &internal_data);
  if (is_carried_at_space) {
    access->io_info = (struct polysa_io_info **)realloc(access->io_info, sizeof(struct polysa_io_info *) * (++access->n_io_info));
    access->io_info[access->n_io_info - 1] = (struct polysa_io_info *)malloc(sizeof(struct polysa_io_info));
    access->io_info[access->n_io_info - 1]->io_type = POLYSA_EXT_IO;
    access->io_info[access->n_io_info - 1]->dep = (struct polysa_dep *)calloc(1, sizeof(struct polysa_dep));
    access->io_info[access->n_io_info - 1]->dep->isl_dep = isl_basic_map_copy(dep);
    access->io_info[access->n_io_info - 1]->dep->type = data->dep_type;
    access->io_info[access->n_io_info - 1]->dir = internal_data.dirvec;
    access->io_info[access->n_io_info - 1]->old_dir = isl_vec_dup(internal_data.dirvec);
  } else {
    access->io_info = (struct polysa_io_info **)realloc(access->io_info, sizeof(struct polysa_io_info *) * (++access->n_io_info));
    access->io_info[access->n_io_info - 1] = (struct polysa_io_info *)malloc(sizeof(struct polysa_io_info));
    access->io_info[access->n_io_info - 1]->io_type = POLYSA_INT_IO;
    access->io_info[access->n_io_info - 1]->dep = (struct polysa_dep *)calloc(1, sizeof(struct polysa_dep));
    access->io_info[access->n_io_info - 1]->dep->isl_dep = isl_basic_map_copy(dep);
    access->io_info[access->n_io_info - 1]->dep->type = data->dep_type;
    /* Assign a default (1,X) direction vector to transfer the data. */    
    access->io_info[access->n_io_info - 1]->dir = internal_data.dirvec; 
    access->io_info[access->n_io_info - 1]->old_dir = isl_vec_dup(internal_data.dirvec); 
//    access->io_info[access->n_io_info - 1]->dir = isl_vec_set_element_si(access->io_info[access->n_io_info - 1]->dir, 0, 1);
  }

  isl_schedule_node_free(node);

  data->is_update = isl_bool_true;

  return isl_stat_ok;
}

isl_bool data_transfer_update_wrap(__isl_keep isl_map *map, void *user)
{
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  for (int i = 0; i < isl_map_n_basic_map(map); i++) {
    isl_basic_map *dep = isl_basic_map_list_get_basic_map(bmap_list, i);
    struct data_transfer_opt_data *opt_data = (struct data_transfer_opt_data *)user;
    data_transfer_update(dep, opt_data);
    isl_basic_map_free(dep);
  }
  isl_basic_map_list_free(bmap_list);
  return isl_bool_true;
}

static isl_stat sa_io_update(struct polysa_kernel *sa) {
  struct polysa_local_array_info *local_array;
  /* Initialize the IO info */
  for (int i = 0; i < sa->n_array; i++) {
    local_array = &sa->array[i];
    for (int j = 0; j < sa->array[i].array->n_ref; j++) {
      struct polysa_stmt_access *access = sa->array[i].array->refs[j];
      access->n_io_info = 0;
      access->io_info = NULL;
    }
    local_array->n_lane = 0;
    local_array->array->n_lane = 0;
  }

  /* Update the IO information */
  for (int i = 0; i < sa->n_array; i++) {
    local_array = &sa->array[i];
    for (int j = 0; j < local_array->array->n_ref; j++) {
      struct polysa_stmt_access *access = local_array->array->refs[j];
      isl_union_map *dep_rar = sa->scop->tagged_dep_rar;
      isl_union_map *dep_flow = sa->scop->tagged_dep_flow;
      isl_union_map *dep_waw = sa->scop->tagged_dep_waw;
      struct data_transfer_opt_data opt_data = {access, sa, POLYSA_DEP_UNKNOWN, isl_bool_false};

      opt_data.dep_type = POLYSA_DEP_RAR;
      isl_union_map_every_map(dep_rar, &data_transfer_update_wrap, &opt_data);
      if (opt_data.is_update == isl_bool_true) {
        local_array->array_type = POLYSA_EXT_ARRAY;
        opt_data.is_update = isl_bool_false;
      }

      opt_data.dep_type = POLYSA_DEP_RAW;
      isl_union_map_every_map(dep_flow, &data_transfer_update_wrap, &opt_data);
      if (opt_data.is_update == isl_bool_true) {
        local_array->array_type = POLYSA_INT_ARRAY;
        opt_data.is_update = isl_bool_false;
      }

      opt_data.dep_type = POLYSA_DEP_WAW;
      isl_union_map_every_map(dep_waw, &data_transfer_update_wrap, &opt_data);     
    }
  }

  return isl_stat_ok;
}

/* Apply data transfer optimization including:
 * - I/O analysis
 * - Interior I/O elimination
 * To transfer data between PEs and external memory, three levels of I/O modules
 * are allocated: L1, L2, L3.
 * L1 modules are located beside PEs and transfer data between PEs and modules.
 * L2 modules are one level above L1. L1 modules are grouped and chained to L2 modules
 * along certain direction.
 * L3 modules are one level above L2. L2 modules are grouped and chained to L1 modules
 * along certain direction.
 * A general topology looks like below for 2D/1D array.
 *
 * DRAM -> L3 module -> L2 module -----------> L2 module 
 *                      |                      |  
 *                      L1 module -> PE        L1 module -> PE
 *                      |                      |
 *                      L1 module -> PE        L1 module -> PE
 *
 * First, I/O analysis is conducted to examine each array access in the kernel.
 * All the array accesses are associated with RAR or RAW dep.
 * If the dep is carried by the space loop, the access is bound with
 * exterior I/O, buffers will be allocated at the L2 level.
 * Else if the dep is carried by the time loop, the access is bound with
 * interior I/O, buffers will be allocated at the L1 level.
 * Each array access is updated the fields of transfer direction and I/O type.
 *
 * For interior I/O, we will assign a default transfer direction (1,0) for transferring 
 * data between PEs and the external memory. Otherwise, global interconnects
 * are introduced for interior I/O which will hurt the timing.
 */
isl_stat sa_data_transfer_optimize(struct polysa_kernel *sa, struct polysa_gen *gen)
{
  printf("[PolySA] Apply data transfer optimization.\n");

  /* Group all the accesses based on the updated IO information */
  sa_group_references(sa, gen);

  return isl_stat_ok;
}

/* Apply PE optimization including:
 * - latency hiding
 * - SIMD vectorization
 * - array partitioning
 */
isl_stat sa_pe_optimize(struct polysa_kernel *sa, bool pass_en[], char *pass_mode[])
{
  printf("[PolySA] Appy PE optimization.\n");
  /* Prepartion before starting the optimization. */
  /* Initialize the polysa_loop_types. */
  sa_loop_init(sa);
  /* Set up the space_time properties. */
  sa_space_time_loop_setup(sa);
  /* Update I/O information */
  sa_io_update(sa);
  /* Extract the tile sizes. */
  sa->sizes = extract_sizes_from_str(sa->ctx, sa->scop->options->sa_sizes);
  /* Set the kernel id. */
  sa->id = 0;
  /* Set the core */
  isl_union_set *domain = isl_schedule_get_domain(sa->schedule);
  sa->core = isl_union_set_universe(domain);

//  // debug
//  isl_printer *p = isl_printer_to_file(sa->ctx, stdout);
//  p = isl_printer_print_union_set(p, sa->core);
//  printf("\n");
//  p = isl_printer_free(p);
//  // debug

  /* Array partitioning. */
  sa_array_partitioning_optimize(sa, pass_en[0], pass_mode[0], pass_en[1], pass_mode[1]);
  /* Latency hiding. */
  sa_latency_hiding_optimize(sa, pass_en[2], pass_mode[2]);
  /* SIMD vectorization. */
  if (pass_en[3])
    sa_simd_vectorization_optimize(sa, pass_mode[3]);

  return isl_stat_ok;
}

static isl_bool update_simd_acc_stmt(__isl_keep isl_set *set, void *user)
{
  struct stride_coalesced_data *data = user;
  struct polysa_stmt *stmt;
  isl_space *space;
  isl_id *id;
  struct polysa_stmt_access *accesses, *access;
  isl_map *prefix;

  space = isl_set_get_space(set);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  isl_space_free(space);
  stmt = find_stmt(data->kernel->prog, id);
  isl_id_free(id);
  accesses = stmt->accesses;
  prefix = isl_map_from_union_map(isl_union_map_intersect_domain(
        isl_union_map_copy(data->prefix), isl_union_set_from_set(isl_set_copy(set))));

  for (access = accesses; access; access = access->next) {
    isl_map *acc;
    int n;
    isl_bool is_zero = isl_bool_false, is_one = isl_bool_false;
    isl_pw_multi_aff *pma;
    int i;

    if (access->n_index == 0)
      continue;

    acc = isl_map_copy(access->access);
    acc = isl_map_apply_domain(acc, isl_map_copy(prefix));

    for (i = access->n_index - 1; i >= 0; i--) {
      is_zero = access_is_stride_zero(acc, i);
      if (is_zero)
        break;
    }
    if (!is_zero) {
      is_one = isl_bool_true;
    }

    isl_map_free(acc);
    access->simd_stride = is_zero? 0 : (is_one? 1 : -1);
  }

  isl_map_free(prefix);
  return isl_bool_true;
}

static isl_bool update_simd_acc(__isl_keep isl_schedule_node *node, void *user)
{
  isl_union_set *domain;
  isl_union_map *prefix;
  struct stride_coalesced_data *data = user;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return isl_bool_true;

  domain = isl_schedule_node_get_domain(node);
  prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
  data->prefix = prefix;

  isl_union_set_every_set(domain, &update_simd_acc_stmt, data);
  
  isl_union_set_free(domain);
  isl_union_map_free(prefix);

  return isl_bool_true;
}

static __isl_give isl_schedule_node *polysa_simd_tile_loop(__isl_take isl_schedule_node *node,
  void *user)
{
  struct simd_vectorization_data *data = user;
  struct polysa_kernel *kernel = data->kernel;
  struct stride_coalesced_data stride_data;
  stride_data.kernel = data->kernel;

//  // debug
//  isl_printer *p = isl_printer_to_file(kernel->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      if (isl_schedule_node_band_member_get_pe_opt(node, i) == polysa_loop_simd) {
        if (!strcmp(data->mode, "auto")) {
          /* Perform tiling on the loop with the highest score. */
          if (data->scores[data->loop_cnt] != data->best_score) { 
            node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
            continue;
          }
        } else {
          /* Peform tiling on the loop with positive tiling factor */
          if (data->tile_size[data->loop_cnt] <= 0) {          
            node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
            continue;
          }
        }
        if (data->tile_size[data->loop_cnt] == 1) {
          /* No meaning to tile when tile size is 1 */
          node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
          continue;
        }
        int tile_size = data->tile_size[data->loop_cnt];
        /* Tile the loop */
        node = polysa_node_band_tile_loop(node, tile_size, i);
        /* Reset the candidate loop in the tile loop the pe_opt property to default */
        node = isl_schedule_node_band_member_set_pe_opt(node, i, polysa_loop_default);
        /* Reset the point loop space_time property to time loop. */
        node = isl_schedule_node_child(node, 0);
        node = isl_schedule_node_band_member_set_space_time(node, 0, polysa_loop_time);
        /* Reset the point loop pe_opt property to default. */
        node = isl_schedule_node_band_member_set_pe_opt(node, 0, polysa_loop_default);
          
        /* Sink the point loop innermost */
        node = isl_schedule_node_band_sink(node);
        /* Add the simd marker */
        node = isl_schedule_node_map_descendant_bottom_up(node, &add_simd_mark, NULL);
        /* Update the array references */
        isl_schedule_node_every_descendant(node, &update_simd_acc, &stride_data); 

        node = isl_schedule_node_parent(node);

//          // debug
//          p = isl_printer_print_schedule_node(p, node);
//          p = isl_printer_flush(p);
//          // debug
           
        kernel->simd_w = tile_size;
        data->loop_cnt++;
      }
    }
  }

  return node;
}

/* Apply SIMD vectorization. 
 * Go through all the loops, if there is any vectorizable loop (parallel or reduction loop
 * with stride-0/1 access), such a loop will be identified as SIMD loop candidate. We will rank
 * the loops by heuristics and pick up one loop to be tiled. The point loops will be permuated 
 * as the innermost loops to be unrolled.
 */
isl_stat sa_simd_vectorization_optimize(struct polysa_kernel *sa, char *mode)
{
  int *scores = NULL;
  int n_loops = 0;
  struct simd_vectorization_data data;
  data.best_score = 0;
  data.mode = mode;
  data.ubs = NULL;
  int *tile_size;

  printf("[PolySA] Apply SIMD vectorization.\n");
  isl_schedule *schedule = sa->schedule; 
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  sa->simd_w = 1;

  /* Move down to the array marker */
  node = polysa_tree_move_down_to_array(node, sa->core);

  /* Detect all candidate loops */
  data.kernel = sa;
  data.scores = scores;
  data.n_loops = n_loops;
  node = isl_schedule_node_map_descendant_bottom_up(
      node, &detect_simd_vectorization_loop, &data);

  if (data.n_loops == 0) {
    printf("[PolySA] No candidate loops found!\n");
    isl_schedule_node_free(node);
    return isl_stat_ok;
  }

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

  if (data.layout_trans) {
    printf("[PolySA] Layout transformation is required to proceed.\n");
    printf("[PolySA] The SIMD vectorization is skipped.\n");
  } else {
    /* Select the candidate loop with the highest score.
    * Tile the candidate loop and permute the point loop innermost. 
    * A SIMD vectorization marker is added. */
    if (!strcmp(mode, "manual")) {
      tile_size = read_simd_tile_sizes(sa, data.n_loops); 
      if (!tile_size) {
        /* Dump out the number, score and upper bounds of simd loops and exit the program. */
        int *ubs = data.ubs;
        FILE *fp;
        char *content;
        cJSON *tuning, *simd_json, *loops_json, *scores_json;
        
        tuning = cJSON_CreateObject();
        simd_json = cJSON_CreateObject();
        cJSON_AddItemToObject(tuning, "simd", simd_json);
        loops_json = cJSON_CreateArray();
        cJSON_AddItemToObject(simd_json, "tilable_loops", loops_json);
        for (int i = 0; i < data.n_loops; i++) {
          cJSON *loop = cJSON_CreateNumber(ubs[i]);
          cJSON_AddItemToArray(loops_json, loop);
        }
        scores_json = cJSON_CreateArray();
        cJSON_AddItemToObject(simd_json, "scores", scores_json);
        for (int i = 0; i < data.n_loops; i++) {
          cJSON *loop = cJSON_CreateNumber(data.scores[i]);
          cJSON_AddItemToArray(scores_json, loop);
        }
        fp = fopen("polysa.tmp/tuning.json", "w");
        content = cJSON_Print(tuning);
        fprintf(fp, "%s", content);
        cJSON_Delete(tuning);
        exit(0);
      }  
    }

    /* Perform the simd vectorization */
    data.loop_cnt = 0;
    data.tile_size = tile_size;
    node = isl_schedule_node_map_descendant_bottom_up(node, 
          &polysa_simd_tile_loop, &data);
  }
  
  free(data.ubs);
  free(tile_size);
  /* Clean up the band pe_opt properties. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  schedule = isl_schedule_map_schedule_node_bottom_up(
      schedule, &clear_pe_opt_prop, NULL);

//  // debug
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  // debug

  free(data.scores);
  sa->schedule = schedule;

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

struct sa_candidates_smart_pick_update_data {
  int score;
  struct polysa_kernel *sa;
  enum polysa_dep_type dep_type;
};

static isl_bool sa_candidates_smart_pick_update(__isl_keep isl_map *map, void *user)
{
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  struct sa_candidates_smart_pick_update_data *data = user;
  struct polysa_kernel *sa = data->sa;
  isl_schedule_node *node = isl_schedule_get_root(sa->schedule);

  for (int i = 0; i < isl_map_n_basic_map(map); i++) {
    isl_basic_map *dep = isl_basic_map_list_get_basic_map(bmap_list, i);
    struct dep_space_test_internal_data internal_data = { NULL, dep };
    isl_bool is_carried_at_space = !isl_schedule_node_every_descendant(node, not_carried_at_space, &internal_data); 
    if (is_carried_at_space && data->dep_type == POLYSA_DEP_RAR)
      data->score += 1;
    else if (!is_carried_at_space && data->dep_type == POLYSA_DEP_RAW)
      data->score += 1;

    isl_vec_free(internal_data.dirvec);  
    isl_basic_map_free(dep);
  }
  isl_schedule_node_free(node);
  isl_basic_map_list_free(bmap_list);
  return isl_bool_true;
}

/* Select one systolic array design based on heuristics. 
 * Heuristic:
 * - RAR carried by space loop. 
 * - RAW carried by time loop. 
 * We compute the score for each design and select the one with the highest score.
 * The score is computed as :
 * score = 1 * (RAR carried by space || RAW carried by time loop)
 * Namely, for each dependnece, if it is a RAR carried by space or a RAW carried by 
 * time loops, it will contriute 1 credit to the total score.
 */
struct polysa_kernel *sa_candidates_smart_pick(struct polysa_kernel **sa_list, __isl_keep isl_size num_sa)
{
  assert(num_sa > 0);
  int max_score = -1;
  struct polysa_kernel *sa_opt;
  int opt_id;
  isl_union_map *dep_rar, *dep_flow;

  for (int i = 0; i < num_sa; i++) {
    struct polysa_kernel *sa = sa_list[i];
    struct sa_candidates_smart_pick_update_data data;
    data.score = 0;
    data.sa = sa;
    /* Initialize the polysa_loop_types. */
    sa_loop_init(sa);
    /* Set up the space_time properties. */
    sa_space_time_loop_setup(sa);

    dep_rar = sa->scop->tagged_dep_rar;
    dep_flow = sa->scop->tagged_dep_flow;
  
    data.dep_type = POLYSA_DEP_RAR;
    isl_union_map_every_map(dep_rar, &sa_candidates_smart_pick_update, &data);
    data.dep_type = POLYSA_DEP_RAW;
    isl_union_map_every_map(dep_flow, &sa_candidates_smart_pick_update, &data);

    if (data.score > max_score) {
      opt_id = i;
      max_score = data.score;
    }
  }

  sa_opt = polysa_kernel_copy(sa_list[opt_id]);
  for (int i = 0; i < num_sa; i++)
    polysa_kernel_free(sa_list[i]);
  free(sa_list);

  return sa_opt;
}

/* Return the selected systolic array design and free the rest. */
struct polysa_kernel *sa_candidates_manual_pick(struct polysa_kernel **sa_list, isl_size num_sa, int sa_id)
{
  struct polysa_kernel *sa_opt = polysa_kernel_copy(sa_list[sa_id]);

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
        isl_schedule *new_schedule = isl_schedule_dup(schedule);       
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
        sa->n_sa_dim = dim;
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
            isl_schedule *new_schedule = isl_schedule_dup(schedule);
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
            sa->n_sa_dim = dim;
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
                isl_schedule *new_schedule = isl_schedule_dup(schedule);
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
                sa->n_sa_dim = dim;
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
        isl_schedule *new_schedule = isl_schedule_dup(schedule);       
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
        sa->n_sa_dim = dim;
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
            isl_schedule *new_schedule = isl_schedule_dup(schedule);
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
            sa->n_sa_dim = dim;
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
                isl_schedule *new_schedule = isl_schedule_dup(schedule);
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
                sa->n_sa_dim = dim;
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

		if (local->n_pe_group == 0)
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

///* If max_local_memory is not set to infinity (-1), then make
// * sure that the total amount of local memory required by the
// * array reference groups mapped to local memory by "kernel"
// * is no larger than this maximum.
// *
// * We apply a greedy approach and discard (keep in global memory)
// * those groups that would result in a total memory size that
// * is larger than the maximum.
// *
// * This function should be called after any function that may
// * affect the decision on whether to place a reference group
// * in local or global memory.
// */
//static void check_local_memory_bound(struct polysa_kernel *kernel)
//{
//	int i, j;
//	isl_val *left, *size;
//
//	if (kernel->options->max_local_memory < 0)
//		return;
//
//	left = isl_val_int_from_si(kernel->ctx,
//				    kernel->options->max_local_memory);
//
//	for (i = 0; i < kernel->n_array; ++i) {
//		struct polysa_local_array_info *local = &kernel->array[i];
//
//		for (j = 0; j < local->n_group; ++j) {
//			struct polysa_array_ref_group *group;
//			enum polysa_group_access_type type;
//
//			group = local->groups[j];
//			type = polysa_array_ref_group_type(group);
//			if (type != POLYSA_ACCESS_LOCAL)
//				continue;
//
//			size = polysa_array_tile_size(group->local_tile);
//			size = isl_val_mul_ui(size, local->array->size); // Byte
//
//			if (isl_val_le(size, left)) {
//				left = isl_val_sub(left, size);
//				continue;
//			}
//			isl_val_free(size);
//
//			group->local_tile =
//					polysa_array_tile_free(group->local_tile);
//		}
//	}
//
//	isl_val_free(left);
//}

///* Mark all arrays of "kernel" that have an array reference group
// * that is not mapped to local memory as
// * accessing the corresponding global device memory.
// */
//static void mark_global_arrays(struct polysa_kernel *kernel)
//{
//	int i, j;
//
//	for (i = 0; i < kernel->n_array; ++i) {
//		struct polysa_local_array_info *local = &kernel->array[i];
//
//		if (local->global)
//			continue;
//		for (j = 0; j < local->n_group; ++j) {
//			if (polysa_array_ref_group_tile(local->groups[j]))
//				continue;
//
//			local->global = 1;
//			local->array->global = 1;
//			break;
//		}
//	}
//}

/* Compute a tiling for all the array reference groups in "kernel".
 */
static void compute_group_tilings_drain(struct polysa_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];
    if (!array->drain_group)
      continue;
		polysa_array_ref_group_compute_tiling(NULL, array->drain_group);
	}
}

/* Compute a tiling for all the array reference groups in "kernel".
 */
static void compute_group_tilings_pe(struct polysa_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_pe_group; ++j)
			polysa_array_ref_group_compute_tiling(NULL, array->pe_groups[j]);
	}
}

/* Compute a tiling for all the array reference groups in "kernel".
 */
static void compute_group_tilings_io(struct polysa_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_io_group; ++j)
			polysa_array_ref_group_compute_tiling(NULL, array->io_groups[j]);
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
static __isl_give isl_union_map *remove_local_accesses_group_flow(
	struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
	__isl_take isl_union_map *access, __isl_keep isl_union_map *prefix,
	int read)
{
	isl_union_map *sched, *tagged;

	if (isl_union_map_is_empty(access))
		return access;

	tagged = group_tagged_access_relation(group);
	sched = isl_union_map_copy(prefix);

	return remove_local_accesses_flow(kernel->prog, tagged, access, sched, read);
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

static __isl_give isl_map *group_tile_buffer(struct polysa_array_ref_group *group,
  struct polysa_array_tile *tile)
{
  int i;
  int n_index = group->array->n_index;
  isl_map *map;
  isl_space *space;
  isl_set *local;
  isl_set *extent;

  space = isl_multi_aff_get_space(tile->tiling);
  space = isl_space_range(space);
  local = isl_set_universe(space);

	for (i = 0; i < n_index; ++i) {
		isl_val *bound;

		local = isl_set_lower_bound_si(local, isl_dim_set, i, 0);
		bound = isl_val_copy(tile->bound[i].size);
		bound = isl_val_sub_ui(bound, 1);
		local = isl_set_upper_bound_val(local, isl_dim_set, i, bound);
	}
	local = isl_set_preimage_multi_aff(local,
				isl_multi_aff_copy(tile->tiling));
	map = isl_set_unwrap(local);
	extent = array_extent(group->array);
	map = isl_map_intersect_range(map, extent);

	return map;
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

  // debug
  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_node(p, node);
  printf("\n");
  // debug

	tile = polysa_array_ref_group_tile(group);
	kernel_depth = isl_schedule_node_get_schedule_depth(node);
	node = polysa_tree_move_down_to_depth(node, tile->depth, kernel->core);

  /* S -> [D -> A] 
   * S: domain elements
   * D: prefix schedule dimensions
   * A: access 
   */
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
    isl_set *set;
    set = isl_map_domain(isl_map_from_union_map(isl_union_set_unwrap(domain)));
//    isl_union_set_free(domain);
    map = group_tile(group);
    map = isl_map_intersect_domain(map, set); 
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

	while (graft && isl_schedule_node_has_parent(graft))
		graft = isl_schedule_node_parent(graft);

	if (read) {
		node = isl_schedule_node_graft_before(node, graft);
	} else {
		node = isl_schedule_node_graft_after(node, graft);
	}

	node = polysa_tree_move_up_to_kernel(node);

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

	type = polysa_cpu_array_ref_group_type(group);
	if (type == POLYSA_ACCESS_LOCAL)
		return add_copies_group_local(kernel, group, node, read); 
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

			type = polysa_cpu_array_ref_group_type(group);
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

			type = polysa_cpu_array_ref_group_type(group);
			if (type == POLYSA_ACCESS_GLOBAL)
				continue;
			create_kernel_var(kernel->ctx, group, &kernel->var[n]);
			++n;
		}
	}

	return isl_stat_ok;
}

/* Compute the effective sa size as a list of the sizes in each dimension.
 *
 * The sa size specified by the user or set by default
 * in read_array_part_tile_sizes() and applied by the PE filter,
 * may be too large for the given code in the sense that
 * it may contain PEs that don't need to execute anything.
 * We therefore don't return this sa size, but instead the
 * smallest grid size that ensures that all blocks that actually
 * execute code are included in the grid.
 *
 * We first extract a description of the grid, i.e., the possible values
 * of the PE ids, from the domain elements in "domain" and
 * kernel->pe_filter.
 * The PE ids are parameters in kernel->pe_filter.
 * We simply need to change them into set dimensions.
 *
 * Then, for each PE dimension, we compute the maximal value of the PE id
 * and add one.
 */
static __isl_give isl_multi_pw_aff *extract_sa_grid_size(
	struct polysa_kernel *kernel, __isl_take isl_union_set *domain)
{
	int i;
	isl_set *grid;
	isl_set *context;
	isl_multi_pw_aff *size;

	domain = isl_union_set_intersect(domain,
				    isl_union_set_copy(kernel->pe_filter));

	grid = isl_union_set_params(domain);
	grid = isl_set_from_params(grid);
	grid = isl_set_add_dims(grid, isl_dim_set, kernel->n_sa_dim);

	for (i = 0; i < kernel->n_sa_dim; ++i) {
		int pos;
		isl_id *id;

		if (!grid)
			return NULL;

		id = isl_id_list_get_id(kernel->pe_ids, i);
		pos = isl_set_find_dim_by_id(grid, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			isl_die(isl_set_get_ctx(grid), isl_error_internal,
				"missing constraints on PE identifier",
				grid = isl_set_free(grid));
		grid = isl_set_equate(grid, isl_dim_param, pos, isl_dim_set, i);
		grid = isl_set_project_out(grid, isl_dim_param, pos, 1);
	}

	grid = isl_set_coalesce(grid);
	size = ppcg_size_from_extent(grid);
	context = isl_set_params(isl_set_copy(kernel->context));
	return isl_multi_pw_aff_gist(size, context);
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
}

static __isl_give isl_schedule_node *sa_add_copies(
    struct polysa_gen *gen, __isl_take isl_schedule_node *node) 
{
  struct polysa_kernel *kernel; 
  isl_id *id;
  isl_set *host_domain;
  isl_union_pw_multi_aff *contraction;
  int single_statement;
  
  id = isl_schedule_node_mark_get_id(node);
  kernel = (struct polysa_kernel *)isl_id_get_user(id);
  host_domain = kernel->host_domain;  
  single_statement = kernel->single_statement;

  /* Add the copy statements. */
  node = add_copies(kernel, node); 

  if (create_kernel_vars(kernel) < 0) 
    node = isl_schedule_node_free(node);

  if (!single_statement)
    node = isl_schedule_node_parent(node);

  isl_id_free(id);

  return node;
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
  /* Enable for array partitioning, L2 array partitioning, latency hiding, SIMD */
  bool pe_opt_en[4]; 
  char *pe_opt_mode[4];
  isl_union_set *domain, *expanded;
  int single_statement;
  isl_union_map *host_schedule;
  isl_set *host_domain;
  isl_id *id;
  isl_union_pw_multi_aff *contraction;
  int n_space_dim;
  char *space_time_mode;
  cJSON *space_time_json, *space_time_mode_json, *n_sa_json, *tuning;
  cJSON *array_part_json, *array_part_en_json, *array_part_mode_json;
  cJSON *array_part_L2_json, *array_part_L2_en_json, *array_part_L2_mode_json;
  cJSON *latency_json, *latency_en_json, *latency_mode_json;
  cJSON *simd_json, *simd_en_json, *simd_mode_json;

  /* Generate systolic arrays using space-time mapping. */
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  sa_candidates = sa_space_time_transform(schedule, gen->prog->scop, &num_sa);
  if (num_sa > 0)
    printf("[PolySA] %d systolic arrays generated.\n", num_sa);
  space_time_json = cJSON_GetObjectItemCaseSensitive(gen->tuning_config, "space_time");
  space_time_mode_json = cJSON_GetObjectItemCaseSensitive(space_time_json, "mode");
  space_time_mode = space_time_mode_json->valuestring;
  if (!strcmp(space_time_mode, "auto")) {
    /* Pick up one systolic array to proceed based on heuristics. */
    kernel = sa_candidates_smart_pick(sa_candidates, num_sa);
  } else {
//    if (gen->options->sa_sizes) 
//      printf("%s\n", gen->options->sa_sizes);
    isl_union_map *sizes = extract_sizes_from_str(gen->ctx, gen->options->sa_sizes);
    int kernel_id = read_space_time_kernel_id(sizes); 
    isl_union_map_free(sizes);
    if (kernel_id < 0) {
      /* Dump out the number of systolic array designs and exit the program*/
      FILE *fp;
      char *content;
      tuning = cJSON_CreateObject();
      space_time_json = cJSON_CreateObject();
      n_sa_json = cJSON_CreateNumber(num_sa);
      cJSON_AddItemToObject(space_time_json, "n_kernel", n_sa_json);
      cJSON_AddItemToObject(tuning, "space_time", space_time_json);
      fp = fopen("polysa.tmp/tuning.json", "w");
      content = cJSON_Print(tuning);
      fprintf(fp, "%s", content);
      cJSON_Delete(tuning);
      exit(0);
    } else {
      kernel = sa_candidates_manual_pick(sa_candidates, num_sa, kernel_id); 
    }
  }
  kernel->prog = gen->prog;
  kernel->options = gen->options;
  /* Create local arrays. */
  kernel = polysa_kernel_create_local_arrays(kernel, gen->prog);

  /* Apply PE optimization. */
  array_part_json = cJSON_GetObjectItemCaseSensitive(gen->tuning_config, "array_part");
  array_part_en_json = cJSON_GetObjectItemCaseSensitive(array_part_json, "enable");
  array_part_mode_json = cJSON_GetObjectItemCaseSensitive(array_part_json, "mode");
 
  array_part_L2_json = cJSON_GetObjectItemCaseSensitive(gen->tuning_config, "array_part_L2");
  array_part_L2_en_json = cJSON_GetObjectItemCaseSensitive(array_part_L2_json, "enable");
  array_part_L2_mode_json = cJSON_GetObjectItemCaseSensitive(array_part_L2_json, "mode");

  latency_json = cJSON_GetObjectItemCaseSensitive(gen->tuning_config, "latency");
  latency_en_json = cJSON_GetObjectItemCaseSensitive(latency_json, "enable");
  latency_mode_json = cJSON_GetObjectItemCaseSensitive(latency_json, "mode");
  
  simd_json = cJSON_GetObjectItemCaseSensitive(gen->tuning_config, "simd");
  simd_en_json = cJSON_GetObjectItemCaseSensitive(simd_json, "enable");
  simd_mode_json = cJSON_GetObjectItemCaseSensitive(simd_json, "mode");

  pe_opt_en[0] = array_part_en_json->valueint;
  pe_opt_en[1] = array_part_L2_en_json->valueint;
  pe_opt_en[2] = latency_en_json->valueint;
  pe_opt_en[3] = simd_en_json->valueint;

  pe_opt_mode[0] = array_part_mode_json->valuestring;
  pe_opt_mode[1] = array_part_L2_mode_json->valuestring;
  pe_opt_mode[2] = latency_mode_json->valuestring;
  pe_opt_mode[3] = simd_mode_json->valuestring;

  sa_pe_optimize(kernel, pe_opt_en, pe_opt_mode);

  /* Create the polysa_kernel object and attach to the schedule. */
  if (!kernel) {
    return NULL;
  }

  node = isl_schedule_get_root(kernel->schedule);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_child(node, 0);

//  // debug
//  isl_printer *p = isl_printer_to_file(gen->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Insert "local" mark before the "array" mark. */
  node = polysa_tree_insert_local_before_array(node);
  if (!node)
    return NULL;

  domain = isl_schedule_node_get_domain(node);
  single_statement = isl_union_set_n_set(domain) == 1;

  /* Prepare some metadata. */
  kernel->single_statement = single_statement;
//  kernel->prog = gen->prog;
  kernel->context = extract_context(node, gen->prog);
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
  host_schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
  host_domain = isl_set_from_union_set(isl_union_map_range(host_schedule));
  kernel->host_domain = host_domain;
  kernel->domain = domain;
  
  /* Make all the host loops atomic so that kernel is only called once. */
  node = atomic_ancestors(node);

  id = isl_id_alloc(gen->ctx, "kernel", kernel);
//  id = isl_id_set_free_user(id, &polysa_kernel_free_wrap);
  node = isl_schedule_node_insert_mark(node, id);
  gen->kernel = kernel;

  if (!single_statement)
    node = group_statements(node, kernel->id); 
  
  /* Insert the PE mark below the space band */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  n_space_dim = 0;
  for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
    if (isl_schedule_node_band_member_get_space_time(node, i) == polysa_loop_space) {
      n_space_dim++;
    }
  }
  node = isl_schedule_node_band_split(node, n_space_dim);
  node = isl_schedule_node_child(node, 0);
  id = isl_id_alloc(gen->ctx, "pe", NULL);
  node = isl_schedule_node_insert_mark(node, id);
  node = polysa_tree_move_up_to_kernel(node);

  /* Save a copy of copy_schedule */
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  kernel->copy_schedule_dim = isl_schedule_node_get_schedule_depth(node);
  kernel->copy_schedule =
    isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
  contraction = isl_union_pw_multi_aff_copy(kernel->contraction);
  kernel->copy_schedule =
    isl_union_pw_multi_aff_pullback_union_pw_multi_aff(
        kernel->copy_schedule, contraction);
  node = polysa_tree_move_up_to_kernel(node);

  /* Delete the local node */
  node = polysa_tree_move_down_to_local(node, kernel->core);
  node = isl_schedule_node_delete(node);

  node = polysa_tree_move_up_to_kernel(node);

  kernel->schedule = isl_schedule_free(kernel->schedule);
  kernel->schedule = isl_schedule_node_get_schedule(node);

  /* Data transfer optimization */
  sa_data_transfer_optimize(kernel, gen);

  /* Localize the array bounds using parameters from the host domain. */
  localize_bounds(kernel, host_domain);

  /* Compute a tiling for all the array reference groups in "kernel". */
  compute_group_tilings_pe(kernel); 
  compute_group_tilings_io(kernel);  
  compute_group_tilings_drain(kernel);

  return node;
}

/* Construct an isl_multi_val for use as tile sizes for tiling "node"
 * from the elements in "tile_size".
 */
static __isl_give isl_multi_val *construct_band_tiles_sizes(
	__isl_keep isl_schedule_node *node, int *tile_size)
{
	isl_space *space;

	if (!node)
		return NULL;

	space = isl_schedule_node_band_get_space(node);
	return ppcg_multi_val_from_int_list(space, tile_size);
}

/* Return an isl_multi_aff, with as elements the parameters in "space"
 * that have the names specified by the elements in "names".
 * If (some of) these parameters do not already appear in "space",
 * then they are added first.
 */
static __isl_give isl_multi_aff *parameter_vector(__isl_take isl_space *space,
	__isl_keep isl_id_list *names)
{
	int i, n;
	isl_local_space *ls;
	isl_multi_aff *ma;

	if (!names)
		space = isl_space_free(space);

	n = isl_id_list_n_id(names);
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;

		id = isl_id_list_get_id(names, i);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		if (pos >= 0) {
			isl_id_free(id);
			continue;
		}
		pos = isl_space_dim(space, isl_dim_param);
		space = isl_space_add_dims(space, isl_dim_param, 1);
		space = isl_space_set_dim_id(space, isl_dim_param, pos, id);
	}
	ma = isl_multi_aff_zero(isl_space_copy(space));
	ls = isl_local_space_from_space(isl_space_domain(space));
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;
		isl_aff *aff;

		id = isl_id_list_get_id(names, i);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
					    isl_dim_param, pos);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_local_space_free(ls);

	return ma;
}

/* Return constraints on the domain elements that equate the partial schedule
 * of "node" to the lower bound of partial schedule. 
 */
static __isl_give isl_union_set *schedule_eq_lb(
  __isl_keep isl_schedule_node *node)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;
  isl_union_map *umap;
  isl_union_set *uset;
  isl_schedule_node *node2;
  isl_bool under_extension = isl_bool_false;

  if (!node)
    return NULL;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  // debug

  /* Test if it is under extension node */
  node2 = isl_schedule_node_copy(node);
  while (node2) {
    if (isl_schedule_node_get_type(node2) == isl_schedule_node_extension) {
      under_extension = isl_bool_true;
      break;
    }
    if (isl_schedule_node_has_parent(node2))
      node2 = isl_schedule_node_parent(node2);
    else 
      break;
//    // debug
//    p = isl_printer_print_schedule_node(p, node2);
//    p = isl_printer_flush(p);
//    // debug
  }
  isl_schedule_node_free(node2);

  
  umap = isl_schedule_node_band_get_partial_schedule_union_map(node);  
//  // debug
//  p = isl_printer_print_union_map(p, umap);
//  printf("\n");
//  // debug
  if (!under_extension) {
    domain = isl_schedule_node_get_domain(node);
    umap = isl_union_map_intersect_domain(umap, domain);
  }
  uset = isl_union_map_range(isl_union_map_copy(umap));
  uset = isl_union_set_lexmin(uset);
  umap = isl_union_map_reverse(umap);
  uset = isl_union_set_apply(uset, umap);

  return uset; 
}

static __isl_give isl_union_set *schedule_neq_lb(
  __isl_keep isl_schedule_node *node)
{
  isl_union_set *uset, *domain;
  isl_union_map *umap;

  if (!node)
    return NULL;

  uset = schedule_eq_lb(node);
  umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
  domain = isl_union_map_domain(umap);
  uset = isl_union_set_subtract(domain, uset);

  return uset;
}

/* Return constraints on the domain elements that equate the partial schedule
 * of "node" to the upper bound of partial schedule. 
 */
static __isl_give isl_union_set *schedule_eq_ub(
  __isl_keep isl_schedule_node *node)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;
  isl_union_map *umap;
  isl_union_set *uset;

  if (!node)
    return NULL;

  domain = isl_schedule_node_get_domain(node);
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  // debug
  umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
  umap = isl_union_map_intersect_domain(umap, domain);
//  // debug
//  p = isl_printer_print_union_map(p, umap);
//  printf("\n");
//  // debug
  uset = isl_union_map_range(isl_union_map_copy(umap));
  uset = isl_union_set_lexmax(uset);
  umap = isl_union_map_reverse(umap);
  uset = isl_union_set_apply(uset, umap);

  return uset;  
}

static __isl_give isl_union_set *schedule_neq_ub(
  __isl_keep isl_schedule_node *node)
{
  isl_union_set *uset, *domain, *sched_domain;
  isl_union_map *umap;

  if (!node)
    return NULL;

  uset = schedule_eq_ub(node);
  domain = isl_schedule_node_get_domain(node);
  umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
  umap = isl_union_map_intersect_domain(umap, domain);
  sched_domain = isl_union_map_domain(umap);
  uset = isl_union_set_subtract(sched_domain, uset);

  return uset;
}

/* Return constraints on the domain elements that equate a sequence of
 * parameters called "names", to the partial schedule of "node".
 * The number of members of the band node "node" should be smaller
 * than or equal to the number of elements in "names". 
 * If it is smaller, then the first elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_eq(
  __isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;

  if (!node)
    return NULL;
  n = isl_id_list_n_id(names);
  if (n == 0)
    return isl_schedule_node_get_universe_domain(node);
  n_zero = n - isl_schedule_node_band_n_member(node);

  mupa = isl_schedule_node_band_get_partial_schedule(node);
  space = isl_multi_union_pw_aff_get_space(mupa);
  space = isl_space_params(space);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, n_zero);
  ma = isl_multi_aff_zero(space);

  domain = isl_schedule_node_get_universe_domain(node);
  /* Map the domain elements to "n_zero" zeros. */
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
            isl_union_set_copy(domain), ma);
  /* Build a new mupa that mupa2 -> mupa */
  mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);  
  space = isl_multi_union_pw_aff_get_space(mupa);
  ma = parameter_vector(space, names);
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
  mupa = isl_multi_union_pw_aff_sub(mupa, mupa2);

  return isl_multi_union_pw_aff_zero_union_set(mupa);
}

/* Return constraints on the domain elements that do not equate a sequence of
 * parameters called "names", to the partial schedule of "node".
 * The number of members of the band node "node" should be smaller
 * than or equal to the number of elements in "names". 
 * If it is smaller, then the first elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_neq(
  __isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;

  if (!node)
    return NULL;
  n = isl_id_list_n_id(names);
  if (n == 0)
    return isl_schedule_node_get_universe_domain(node);
  n_zero = n - isl_schedule_node_band_n_member(node);

  mupa = isl_schedule_node_band_get_partial_schedule(node);
  space = isl_multi_union_pw_aff_get_space(mupa);
  space = isl_space_params(space);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, n_zero);
  ma = isl_multi_aff_zero(space);

  domain = isl_schedule_node_get_universe_domain(node);
  /* Map the domain elements to "n_zero" zeros. */
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
            isl_union_set_copy(domain), ma);
  /* Build a new mupa that mupa2 -> mupa */
  mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);  
  space = isl_multi_union_pw_aff_get_space(mupa);
  ma = parameter_vector(space, names);
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
  mupa = isl_multi_union_pw_aff_sub(mupa, mupa2);

  return isl_multi_union_pw_aff_non_zero_union_set(mupa);
}


/* Return constraints on the domain elements that greater or equal to a sequence of
 * parameters called "names", to the partial schedule of "node".
 * The number of members of the band node "node" should be smaller
 * than or equal to the number of elements in "names". 
 * If it is smaller, then the first elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_ge(
  __isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;

  if (!node)
    return NULL;
  n = isl_id_list_n_id(names);
  if (n == 0)
    return isl_schedule_node_get_universe_domain(node);
  n_zero = n - isl_schedule_node_band_n_member(node);

  mupa = isl_schedule_node_band_get_partial_schedule(node);
  space = isl_multi_union_pw_aff_get_space(mupa);
  space = isl_space_params(space);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, n_zero);
  ma = isl_multi_aff_zero(space);
  domain = isl_schedule_node_get_universe_domain(node);
  /* Generate the mupa that is on the same domain of partial schedule, with
   * a function that maps to the n_zero dims to zero. */
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
            isl_union_set_copy(domain), ma);
  
  /* Generate the mupa with the n_zero dims as paramters and equal zero. */
  mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);  
  space = isl_multi_union_pw_aff_get_space(mupa);
  ma = parameter_vector(space, names);
  /* Generate the mupa that is on the same domain of partial schedule, with
   * a function that maps the domain elements to the parameters. */ 
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
  mupa = isl_multi_union_pw_aff_sub(mupa, mupa2);

  return isl_multi_union_pw_aff_nonneg_union_set(mupa);
}

/* Return constraints on the domain elements that less or equal to a sequence of
 * parameters called "names", to the partial schedule of "node".
 * The number of members of the band node "node" should be smaller
 * than or equal to the number of elements in "names". 
 * If it is smaller, then the first elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_le(
  __isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names)
{
  int n, n_zero;
  isl_multi_union_pw_aff *mupa, *mupa2;
  isl_multi_aff *ma;
  isl_space *space;
  isl_union_set *domain;

  if (!node)
    return NULL;
  n = isl_id_list_n_id(names);
  if (n == 0)
    return isl_schedule_node_get_universe_domain(node);
  n_zero = n - isl_schedule_node_band_n_member(node);

  mupa = isl_schedule_node_band_get_partial_schedule(node);
  space = isl_multi_union_pw_aff_get_space(mupa);
  space = isl_space_params(space);
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, n_zero);
  ma = isl_multi_aff_zero(space);
  domain = isl_schedule_node_get_universe_domain(node);
  /* Generate the mupa that is on the same domain of partial schedule, with
   * a function that maps to the n_zero dims to zero. */
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
            isl_union_set_copy(domain), ma);
  
  /* Generate the mupa with the n_zero dims as paramters and equal zero. */
  mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);  
  space = isl_multi_union_pw_aff_get_space(mupa);
  ma = parameter_vector(space, names);
  /* Generate the mupa that is on the same domain of partial schedule, with
   * a function that maps the domain elements to the parameters. */ 
  mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
  mupa = isl_multi_union_pw_aff_sub(mupa2, mupa);

  return isl_multi_union_pw_aff_nonneg_union_set(mupa);
}

/* Return constraints on the domain elements that equate a sequence of
 * parameters called "names", to the partial schedule
 * of "node" modulo the integers in "size".
 * The number of elements in the array "size" should be equal
 * to the number of elements in "names".
 * The number of members of the band node "node" should be smaller
 * than or equal to this number.  If it is smaller, then the first
 * elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_modulo(
	__isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names,
	int *size)
{
	int n, n_zero;
	isl_space *space;
	isl_multi_aff *ma;
	isl_multi_union_pw_aff *mupa, *mupa2;
	isl_multi_val *mv;
	isl_union_set *domain;

	if (!node)
		return NULL;
	n = isl_id_list_n_id(names);
	if (n == 0)
		return isl_schedule_node_get_universe_domain(node);
	n_zero = n - isl_schedule_node_band_n_member(node);

	mupa = isl_schedule_node_band_get_partial_schedule(node);
	mv = construct_band_tiles_sizes(node, size + n_zero);
	mupa = isl_multi_union_pw_aff_mod_multi_val(mupa, mv);
//  // debug
//  printf("%d %d\n", size[0], size[1]);
//  isl_printer *printer = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  isl_printer_print_multi_val(printer, mv);
//  printf("\n");
//  isl_printer_print_multi_union_pw_aff(printer, mupa);
//  printf("\n");
//  // debug

	space = isl_multi_union_pw_aff_get_space(mupa);
	space = isl_space_params(space);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, n_zero);
	ma = isl_multi_aff_zero(space);

	domain = isl_schedule_node_get_universe_domain(node);
	mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
						isl_union_set_copy(domain), ma);
	mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);

	space = isl_multi_union_pw_aff_get_space(mupa);
	ma = parameter_vector(space, names);

	mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
	mupa = isl_multi_union_pw_aff_sub(mupa, mupa2);

	return isl_multi_union_pw_aff_zero_union_set(mupa);
}

/* If the band node "node" has more than "n" members, then split off
 * the first "n" of them.
 */
static __isl_give isl_schedule_node *split_band(
	__isl_take isl_schedule_node *node, int n)
{
	int dim;

	dim = isl_schedule_node_band_n_member(node);
	if (n < dim)
		node = isl_schedule_node_band_split(node, n);

	return node;
}

/* Add "len" parameters p[i] with identifiers "ids" and intersect "set"
 * with
 *
 *	{ : 0 <= p[i] < size[i] }
 *
 * or an overapproximation.
 */
static __isl_give isl_set *add_bounded_parameters_dynamic(
	__isl_take isl_set *set, __isl_keep isl_multi_pw_aff *size,
	__isl_keep isl_id_list *ids)
{
	int i, len;
	unsigned nparam;
	isl_space *space;
	isl_local_space *ls;

	len = isl_multi_pw_aff_dim(size, isl_dim_out);
	nparam = isl_set_dim(set, isl_dim_param);
	set = isl_set_add_dims(set, isl_dim_param, len);

	for (i = 0; i < len; ++i) {
		isl_id *id;

		id = isl_id_list_get_id(ids, i);
		set = isl_set_set_dim_id(set, isl_dim_param, nparam + i, id);
	}

	space = isl_space_params(isl_set_get_space(set));
	ls = isl_local_space_from_space(space);
	for (i = 0; i < len; ++i) {
		isl_pw_aff *param, *size_i, *zero;
		isl_set *bound;

		param = isl_pw_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_param, nparam + i);

		size_i = isl_multi_pw_aff_get_pw_aff(size, i);
		bound = isl_pw_aff_lt_set(isl_pw_aff_copy(param), size_i);
		bound = isl_set_from_basic_set(isl_set_simple_hull(bound));
		set = isl_set_intersect_params(set, bound);

		zero = isl_pw_aff_zero_on_domain(isl_local_space_copy(ls));
		bound = isl_pw_aff_ge_set(param, zero);
		set = isl_set_intersect_params(set, bound);
	}
	isl_local_space_free(ls);

	return set;
}

/* Insert a context node at "node" introducing the PE identifiers 
 * along with their bounds, which are stored in kernel->sa_grid_size.
 */
static __isl_give isl_schedule_node *insert_context(struct polysa_kernel *kernel,
	__isl_take isl_schedule_node *node)
{
	isl_set *context;

	context = isl_set_universe(isl_set_get_space(kernel->context));

	context = add_bounded_parameters_dynamic(context,
					kernel->sa_grid_size, kernel->pe_ids);
//	context = add_bounded_parameters(context,
//					kernel->block_dim, kernel->thread_ids);

	node = isl_schedule_node_insert_context(node, context);

	return node;
}

static __isl_give isl_union_map *io_comm_access(
  struct polysa_kernel *kernel, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, int read)
{
  isl_union_map *prefix;
  isl_union_map *access;

  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction));
  access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    if (group->group_type == POLYSA_IO_GROUP)
      access = isl_union_map_union(access, polysa_io_group_ref_access_relation(
            group, ref, read, !read));
    else if (group->group_type == POLYSA_DRAIN_GROUP)
      access = isl_union_map_union(access, polysa_drain_group_ref_access_relation(
            group, ref, read, !read, kernel->expanded_domain));
  }

  if (group->local_array->array_type == POLYSA_INT_ARRAY)
    access = remove_local_accesses_group_flow(kernel, group, access, prefix, read);

  access = isl_union_map_range_product(prefix, access);

  return access;
}

/* For:
 * - read access with RAW, project out input dims
 * - write access with RAW, project out output dims
 * Map the domain elements in the access relations to the outer
 * scheduling dimensions (depth above PE level)
 * If the array is an internal array (with RAW), remove the local accesses.
 * Then return the union of these relations.
 */
static __isl_give isl_union_map *pe_int_comm_access(
  struct polysa_kernel *kernel, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, int read)
{
  isl_union_map *prefix;
  isl_union_map *access;

  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction));
  access = polysa_io_group_access_relation(group, read, !read);

  if (group->local_array->array_type == POLYSA_INT_ARRAY)
    access = remove_local_accesses_group_flow(kernel, group, access, prefix, read);

  /* Prefix: D -> S
   * Access: D -> A
   * Range product: D -> [S -> A]
   */
  access = isl_union_map_range_product(prefix, access);
  
  return access;
}

static __isl_give isl_union_map *pe_ext_comm_access(
  struct polysa_kernel *kernel,  
  __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group,    
  struct polysa_stmt_access *ref, int read)
{
  isl_union_map *prefix;
  isl_union_map *access;  
  isl_map *access_i;

  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction)); 
  if (group->group_type == POLYSA_IO_GROUP)
    access = polysa_io_group_ref_access_relation(group, ref, read, !read);
  else if (group->group_type == POLYSA_DRAIN_GROUP)
    access = polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain);

  if (group->local_array->array_type == POLYSA_INT_ARRAY)
    access = remove_local_accesses_group_flow(kernel, group, access, prefix, read);
  
  access = isl_union_map_range_product(prefix, access);
  
  return access;  
}

static __isl_give isl_union_map *pe_drain_access(
  struct polysa_kernel *kernel, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, int read)
{
  isl_union_map *prefix;
  isl_union_map *access;

  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction));
  access = polysa_drain_group_access_relation(group, read, !read, kernel->expanded_domain);

  access = isl_union_map_range_product(prefix, access);

  return access;
}

/* Given an array reference group "group", create a mapping
 *
 * read.suffix[D -> A] -> [D -> A]
 *
 * if "read" is set or 
 *
 * write.suffix[D -> A] -> [D -> A]
 *
 * if "read" is not set.
 * D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 *
 * "local group" marks the local array references. If it is mapped 
 * to registers, the "local group" is set NULL.
 * "tile" is the tile to be copied.
 */
static __isl_give isl_multi_aff *polysa_create_rw_access_suffix(
  isl_ctx *ctx, 
  struct polysa_array_ref_group *local_group, 
  struct polysa_array_ref_group *io_group,
  struct polysa_array_tile *tile,
  int read, char *suffix)
{
  isl_space *space;
  isl_id *id;
  char *str;
  char buf[50];
  struct polysa_array_ref_group_pair *pair = 
    (struct polysa_array_ref_group_pair *)malloc(sizeof(struct polysa_array_ref_group_pair));
  pair->local_group = local_group;
  pair->io_group = io_group;
  pair->in_use = 0;

  if (tile == NULL)
    tile = polysa_array_ref_group_tile(local_group);
  space = isl_space_copy(io_group->array->space);
  space = isl_space_from_range(space);
  space = isl_space_add_dims(space, isl_dim_in, tile->depth);
  space = isl_space_wrap(space);
  space = isl_space_map_from_set(space);
  
  isl_printer *p_str = isl_printer_to_str(ctx);
  if (read)
    p_str = isl_printer_print_str(p_str, "read");
  else
    p_str = isl_printer_print_str(p_str, "write");
  if (suffix) {
    p_str = isl_printer_print_str(p_str, ".");
    p_str = isl_printer_print_str(p_str, suffix);
  }
  str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  sprintf(buf, "%s", str);
  free(str);

  id = isl_id_alloc(ctx, buf, pair);
  id = isl_id_set_free_user(id, &free_group_pair);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);

  return isl_multi_aff_identity(space);
}

/* Given an array reference group "group", create a mapping
 *
 * in.suffix[D -> A] -> [D -> A]
 *
 * if "in" is set or 
 *
 * out.suffix[D -> A] -> [D -> A]
 *
 * if "in" is not set.
 *
 * create the mapping
 *
 * in_reg.suffix[D -> A] -> [D -> A]
 *
 * or 
 *
 * out_reg.suffix[D -> A] -> [D -> A]
 *
 * if "reg" is set.
 *
 * D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 *
 * "local group" marks the local array references. If it is mapped 
 * to registers, the "local group" is set NULL.
 * "tile" is the tile to be copied.
 */
static __isl_give isl_multi_aff *polysa_create_io_access_suffix(
  isl_ctx *ctx, 
  struct polysa_array_ref_group *local_group, 
  struct polysa_array_ref_group *io_group,
  struct polysa_array_tile *tile,
  int in,
  int reg,
  __isl_keep char *suffix)
{
  isl_space *space;
  isl_id *id;
  char *str;
  char buf[50];
  struct polysa_array_ref_group_pair *pair = 
    (struct polysa_array_ref_group_pair *)malloc(sizeof(struct polysa_array_ref_group_pair));
  pair->local_group = local_group;
  pair->io_group = io_group;
  pair->in_use = 0;

  if (tile == NULL)
    tile = polysa_array_ref_group_tile(local_group);
  space = isl_space_copy(io_group->array->space);
  space = isl_space_from_range(space);
  space = isl_space_add_dims(space, isl_dim_in, tile->depth);
  space = isl_space_wrap(space);
  space = isl_space_map_from_set(space);
  
  isl_printer *p_str = isl_printer_to_str(ctx);
  if (in)
    p_str = isl_printer_print_str(p_str, "in");
  else
    p_str = isl_printer_print_str(p_str, "out");
  if (reg) {
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, "reg");
  }
  if (suffix) {
    p_str = isl_printer_print_str(p_str, ".");
    p_str = isl_printer_print_str(p_str, suffix);
  }
  str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  sprintf(buf, "%s", str);
  free(str);

  id = isl_id_alloc(ctx, buf, pair);
  id = isl_id_set_free_user(id, &free_group_pair);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);

  return isl_multi_aff_identity(space);
}

/* "io_group" is the current I/O group that is analyzed.
 * "local_tile" is the tile that the current io stmt accesses.
 * "depth" is the schedule depth that the current stmt is inserted at.
 */
static __isl_give isl_multi_aff *polysa_create_io_access_stmt(
  isl_ctx *ctx,
  struct polysa_array_ref_group *local_group,
  struct polysa_array_ref_group *io_group,
  struct polysa_array_tile *tile,
  int depth,
  __isl_keep char *stmt_name)
{
  isl_space *space;
  isl_id *id;  
  char buf[100];
  struct polysa_array_ref_group_pair *pair = 
    (struct polysa_array_ref_group_pair *)malloc(sizeof(struct polysa_array_ref_group_pair));
  pair->local_group = local_group;
  pair->io_group = io_group;
  pair->local_tile = tile;
  pair->in_use = 0;

  space = isl_space_copy(io_group->array->space);
  space = isl_space_from_range(space);
  space = isl_space_add_dims(space, isl_dim_in, depth);
  space = isl_space_wrap(space);
  space = isl_space_map_from_set(space);
  
  sprintf(buf, "%s", stmt_name);

  id = isl_id_alloc(ctx, buf, pair);
  id = isl_id_set_free_user(id, &free_group_pair);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);

  return isl_multi_aff_identity(space);
}


/* Given an array reference group "group", create a mapping
 *
 * read.fifoX[D -> A] -> [D -> A]
 *
 * if "read" is set or 
 *
 * write.fifoX[D -> A] -> [D -> A]
 *
 * if "read" is not set.
 * D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 */
static __isl_give isl_multi_aff *polysa_create_rw_access(isl_ctx *ctx, 
  struct polysa_array_ref_group *local_group, 
  struct polysa_array_ref_group *io_group,
  struct polysa_array_tile *tile, 
  int read)
{
  isl_space *space;
  isl_id *id;
  char *str;
  char buf[50];
  struct polysa_array_ref_group_pair *pair = 
    (struct polysa_array_ref_group_pair *)malloc(sizeof(struct polysa_array_ref_group_pair));
  pair->local_group = local_group;
  pair->io_group = io_group;
  pair->in_use = 0;

  if (tile == NULL)
    tile = polysa_array_ref_group_tile(local_group);
  space = isl_space_copy(io_group->array->space);
  space = isl_space_from_range(space);
  space = isl_space_add_dims(space, isl_dim_in, tile->depth);
  space = isl_space_wrap(space);
  space = isl_space_map_from_set(space);
  
  isl_printer *p_str = isl_printer_to_str(ctx);
  if (read)
    p_str = isl_printer_print_str(p_str, "read");
  else
    p_str = isl_printer_print_str(p_str, "write");
  p_str = isl_printer_print_str(p_str, ".");
  if (io_group->group_type != POLYSA_PE_GROUP) {
    p_str = isl_printer_print_str(p_str, "fifo_");
  }
  p_str = isl_printer_print_str(p_str, io_group->array->name);
  if (io_group->group_type == POLYSA_IO_GROUP) {
    if (io_group->local_array->n_io_group > 1) {
      p_str = isl_printer_print_str(p_str, "_");
      p_str = isl_printer_print_int(p_str, io_group->nr);
    }
  } else if (io_group->group_type == POLYSA_DRAIN_GROUP) {
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, "drain");
  }
//  if (read) {
//    p_str = isl_printer_print_str(p_str, "_in");
//  } else {
//    p_str = isl_printer_print_str(p_str, "_out");
//  }
  str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  sprintf(buf, "%s", str);
  free(str);

  id = isl_id_alloc(ctx, buf, pair);
  id = isl_id_set_free_user(id, &free_group_pair);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);

  return isl_multi_aff_identity(space);
}

/* Given an array reference group "group", create a mapping
 *
 * in.fifoX[D -> A] -> [D -> A]
 *
 * if "read" is set or 
 *
 * out.fifoX[D -> A] -> [D -> A]
 *
 * if "in" is not set.
 * D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 */
static __isl_give isl_multi_aff *polysa_create_io_access(isl_ctx *ctx,
  struct polysa_array_ref_group *local_group,
  struct polysa_array_ref_group *io_group,
  struct polysa_array_tile *tile, 
  int depth,
  int in)
{
  isl_space *space;
  isl_id *id;
  char *str;
  char buf[50];
  struct polysa_array_ref_group_pair *pair = 
    (struct polysa_array_ref_group_pair *)malloc(sizeof(struct polysa_array_ref_group_pair));
  pair->local_group = local_group;
  pair->io_group = io_group;
  pair->local_tile = tile;
  pair->in_use = 0;

  space = isl_space_copy(io_group->array->space);
  space = isl_space_from_range(space);
  space = isl_space_add_dims(space, isl_dim_in, depth);
  space = isl_space_wrap(space);
  space = isl_space_map_from_set(space);
  
  isl_printer *p_str = isl_printer_to_str(ctx);
  if (in)
    p_str = isl_printer_print_str(p_str, "in");
  else
    p_str = isl_printer_print_str(p_str, "out");
  p_str = isl_printer_print_str(p_str, ".");
  if (io_group->group_type != POLYSA_PE_GROUP) {
    p_str = isl_printer_print_str(p_str, "fifo_");
  }
  p_str = isl_printer_print_str(p_str, io_group->array->name);
  if (io_group->group_type == POLYSA_IO_GROUP) {
    if (io_group->local_array->n_io_group > 1) {
      p_str = isl_printer_print_str(p_str, "_");
      p_str = isl_printer_print_int(p_str, io_group->nr);
    }
  } else if (io_group->group_type == POLYSA_DRAIN_GROUP) {
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, "drain");
  }
  // TODO
  p_str = isl_printer_print_str(p_str, ".");
  p_str = isl_printer_print_int(p_str, io_group->n_lane);
  p_str = isl_printer_print_str(p_str, ".");
  p_str = isl_printer_print_int(p_str, io_group->n_lane);

//  if (read) {
//    p_str = isl_printer_print_str(p_str, "_in");
//  } else {
//    p_str = isl_printer_print_str(p_str, "_out");
//  }
  str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  sprintf(buf, "%s", str);
  free(str);

  id = isl_id_alloc(ctx, buf, pair);
  id = isl_id_set_free_user(id, &free_group_pair);
  space = isl_space_set_tuple_id(space, isl_dim_in, id);

  return isl_multi_aff_identity(space);
}

struct polysa_add_pe_ext_io_copies_data {
  struct polysa_kernel *kernel;
  struct polysa_array_ref_group *pe_group;
  struct polysa_array_ref_group *io_group;
  struct polysa_stmt_access *ref;
  int read;
  int dummy;
  isl_union_set *filter;
};

static __isl_give isl_union_map *io_comm_access_ref(
  struct polysa_kernel *kernel, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, 
  struct polysa_stmt_access *ref,
  int read)
{
  isl_union_map *prefix;
  isl_union_map *access;

  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction));
  if (group->group_type == POLYSA_IO_GROUP)
    access = polysa_io_group_ref_access_relation(group, ref, read, !read);
  else if (group->group_type == POLYSA_DRAIN_GROUP)
    access = polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain);

  if (group->local_array->array_type == POLYSA_INT_ARRAY)
    access = remove_local_accesses_group_flow(kernel, group, access, prefix, read);

  access = isl_union_map_range_product(prefix, access);

  return access;
}

static struct polysa_array_tile *create_register_tiling(
  isl_schedule_node *node,
  struct polysa_array_ref_group *group,
  struct polysa_stmt_access *ref)
{
  isl_union_map *access;
  isl_map *access_i;
  isl_ctx *ctx;
  isl_union_map *sched;
  isl_bool ok;
  struct polysa_array_tile *tile;
  
  ctx = isl_schedule_node_get_ctx(node);
  access = isl_union_map_from_map(isl_map_copy(ref->access)); 
  tile = polysa_array_tile_create(ctx, group->array->n_index); 
  sched = isl_schedule_node_get_prefix_schedule_union_map(node); 
  access = isl_union_map_apply_domain(access, sched); 
  access_i = isl_map_from_union_map(access); 
  ok = can_tile(access_i, tile);
  
  isl_map_free(access_i);

  polysa_array_ref_group_compute_tiling(tile, group);

  return tile;
}

/* Given a schedule node "node" of the type "isl_schedule_node_leaf", 
 * we will test if it is under any extension node.
 * If so, we will then test if the current node intersect with the extension domain. 
 */
static isl_bool leaf_node_is_extended(__isl_keep isl_schedule_node *node)
{
  isl_schedule_node *node_e;
  isl_schedule_node *node_f;
  isl_union_set *filter;
  isl_union_map *extension;
  isl_union_set *extension_range;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return isl_bool_error;

  node_e = isl_schedule_node_copy(node); 
  node_f = isl_schedule_node_copy(node); 

  while (node_e && isl_schedule_node_has_parent(node_e)) {
    if (isl_schedule_node_get_type(node_e) == isl_schedule_node_extension)
      break;
    node_e = isl_schedule_node_parent(node_e);
  }

  if (node_e == NULL || isl_schedule_node_get_type(node_e) != isl_schedule_node_extension) {
    isl_schedule_node_free(node_e);
    isl_schedule_node_free(node_f);
    return isl_bool_false;
  }

  extension = isl_schedule_node_extension_get_extension(node_e); 

  while (node_f && isl_schedule_node_has_parent(node_f)) {
    if (isl_schedule_node_get_type(node_f) == isl_schedule_node_filter)
      break;
    node_f = isl_schedule_node_parent(node_f);
  }

  filter = isl_schedule_node_filter_get_filter(node_f); 
  extension_range = isl_union_map_range(extension); 
  filter = isl_union_set_intersect(filter, extension_range); 
  isl_schedule_node_free(node_e);
  isl_schedule_node_free(node_f);
  if (isl_union_set_is_empty(filter)) {
    isl_union_set_free(filter);
    return isl_bool_false;
  }
  
  isl_union_set_free(filter);
  return isl_bool_true;
}

/* Insert data transfer statements beside the program statements. 
 * If the statement is under the SIMD loop, the data transfer statements are inserted 
 * before/after the SIMD loop. 
 * Otherwise, it is inserted before/after the statement.
 */
__isl_give isl_schedule_node *add_pe_ext_io_copies_stmt(__isl_take isl_schedule_node *node, void *user)
{
  struct polysa_add_pe_ext_io_copies_data *data = (struct polysa_add_pe_ext_io_copies_data *)(user);
  isl_union_set *domain;
  isl_space *space;
  isl_space *acc_space;
  isl_id *id;
  isl_union_map *access;
  int empty;
  isl_multi_aff *from_access;
  isl_ctx *ctx;
  isl_schedule_node *graft;
  isl_multi_aff *ma;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  struct polysa_array_ref_group *pe_group = data->pe_group;
  struct polysa_array_ref_group *io_group = data->io_group;
  struct polysa_array_tile *tile;
  int read = data->read; 
  isl_union_map *sched;
  isl_union_map *ref_access;
  isl_map *acc;
  isl_bool ok;
  int is_simd;
  isl_printer *p_str;
  char *stmt_name;
  isl_union_set *empty_filter;
  int n_lane = io_group->n_lane;

  /* Debug */
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
  /* Debug */

  /* Test if the current stmt contains the reference. */
  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  /* Test if the node is under any extension node and if the 
   * node is extended by the extension node. 
   */
  if (!leaf_node_is_extended(node)) {
//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
    isl_set *set;
    isl_id *new_id;
    domain = isl_schedule_node_get_domain(node); 
    set = isl_set_from_union_set(domain); 
    space = isl_set_get_space(set); 
    isl_set_free(set);
    id = isl_space_get_tuple_id(space, isl_dim_set); 
    isl_space_free(space);
    acc_space = isl_map_get_space(data->ref->access); 
    new_id = isl_space_get_tuple_id(acc_space, isl_dim_in);
    if (id != new_id) {
      isl_space_free(acc_space);
      isl_id_free(id);
      isl_id_free(new_id);
      
      /* Insert empty filter for dummy module */
      if (data->dummy) {
        empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(data->kernel->context)));
        node = isl_schedule_node_insert_filter(node, empty_filter);
      }
      return node;
    }
    isl_id_free(id);
    isl_id_free(new_id);
    isl_space_free(acc_space);
  } else {
//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
    return node;
  }

  ctx = isl_schedule_node_get_ctx(node);
  tile = NULL;
  /* Examine if there is any SIMD mark above */
  is_simd = is_node_under_simd(node); 

  /* Aggregate the copy-in/out access
   * S -> [D -> A]
   * S: statement domain elements
   * D: prefix schedule dimensions
   * A: access
   */
//  access = pe_ext_comm_access(data->kernel, node, io_group, data->ref, read);
  if (is_simd) {
    if (data->dummy) {
      isl_union_set *empty_filter;
      empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(data->kernel->context)));
      node = isl_schedule_node_insert_filter(node, empty_filter);
    }
    node = polysa_tree_move_up_to_mark(node, "simd");
  }
  access = io_comm_access_ref(data->kernel, node, io_group, data->ref, read);
  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return polysa_tree_move_up_to_kernel(node);
  }

  if (data->dummy) {
    data->filter = isl_schedule_node_get_domain(node);
  }

  /* Update the group io_dir */
  if (!data->dummy) {
    if (read) {
      io_group->pe_io_dir = (io_group->pe_io_dir == IO_OUT)? IO_INOUT : IO_IN; 
    } else {
      io_group->pe_io_dir = (io_group->pe_io_dir == IO_IN)? IO_INOUT : IO_OUT;
    }
  }

  pe_group->array->global = 1;
  pe_group->local_array->global = 1;

  /* read.fifoX[D -> A] -> [D -> A] */
  p_str = isl_printer_to_str(ctx);
  if (read)
    p_str = isl_printer_print_str(p_str, "in");
  else
    p_str = isl_printer_print_str(p_str, "out");
  if (data->dummy)
    p_str = isl_printer_print_str(p_str, "_dummy");
  p_str = isl_printer_print_str(p_str, ".");
  if (io_group->group_type != POLYSA_PE_GROUP) {
    p_str = isl_printer_print_str(p_str, "fifo_");
  }
  p_str = isl_printer_print_str(p_str, io_group->array->name);
  if (io_group->group_type == POLYSA_IO_GROUP) {
    if (io_group->local_array->n_io_group > 1) {
      p_str = isl_printer_print_str(p_str, "_");
      p_str = isl_printer_print_int(p_str, io_group->nr);
    }
  } else if (io_group->group_type == POLYSA_DRAIN_GROUP) {
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, "drain");
  }
  p_str = isl_printer_print_str(p_str, ".");
  p_str = isl_printer_print_int(p_str, io_group->n_lane);
  p_str = isl_printer_print_str(p_str, ".1");
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
    
  from_access = polysa_create_io_access_stmt(ctx, pe_group, io_group, 
    polysa_array_ref_group_tile(pe_group),
    isl_schedule_node_get_schedule_depth(node), stmt_name);
  free(stmt_name);

//  from_access = polysa_create_io_access(ctx, pe_group, io_group, polysa_array_ref_group_tile(pe_group), isl_schedule_node_get_schedule_depth(node), read); 

  /* Create a register tiling. */
  tile = create_register_tiling(node, pe_group, data->ref); 
  /* [D -> A] -> T */
  ma = isl_multi_aff_copy(tile->tiling);
  ma = isl_multi_aff_pullback_multi_aff(ma,
      isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma); 

  /* read.fifoX[D -> A] -> T */
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa); 

  /* [D -> A] */
  domain = isl_union_map_range(access); 
  
  /* read.fifoX[D -> A] */
  domain = isl_union_set_preimage_multi_aff(domain, from_access); 
  /* read.fifoX[D -> A] -> D */
  access = isl_union_set_wrapped_domain_map(domain); 
  /* D -> read.fifoX[D -> A] */
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  if (n_lane > 1) {
    /* Perform data packing */
    int n_index;
    int tile_size[1];
    isl_id *id;
    isl_union_map *umap;
    isl_union_set *filter;

    n_index = isl_schedule_node_band_n_member(graft);
    /* Split off the last dimension */
    graft = isl_schedule_node_band_split(graft, n_index - 1);
    graft = isl_schedule_node_child(graft, 0);
    /* Tile the last dimension */
    tile_size[0] = n_lane;
    graft = polysa_tile_band(graft, tile_size);
    graft = isl_schedule_node_child(graft, 0);
    /* Create a filter */
    filter = schedule_eq_lb(graft);
    graft = isl_schedule_node_insert_filter(graft, filter);
  }

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

//  if (is_simd) {
//    node = polysa_tree_move_up_to_mark(node, "simd");
//  }

  if (read) {
    node = isl_schedule_node_graft_before(node, graft);
  } else {
    node = isl_schedule_node_graft_after(node, graft);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  if (data->dummy) {
    /* insert an empty filter */
    empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(data->kernel->context)));
    node = isl_schedule_node_insert_filter(node, empty_filter);
  }

  node = isl_schedule_node_parent(node); // filter
  node = isl_schedule_node_parent(node); // sequence
  node = isl_schedule_node_parent(node); // extension

  polysa_array_tile_free(tile);

  return node;
}

///* Add the statements for copy-out the final results.
// * The "node" is pointed to the "PE" mark.
// */
//__isl_give isl_schedule_node *add_pe_drain_copies(
//  struct polysa_kernel *kernel,
//  struct polysa_local_array_info *local_array,
//  struct polysa_array_ref_group *drain_group,
//  __isl_take isl_schedule_node *node, int read)
//{
//  struct polysa_array_tile *tile;
//  isl_union_map *access;
//  int empty;
//  isl_multi_aff *from_access;
//  isl_multi_aff *ma;
//  isl_multi_pw_aff *mpa;
//  isl_multi_union_pw_aff *mupa;
//  isl_union_set *domain;
//  isl_schedule_node *graft;
//  struct polysa_array_ref_group *pe_group;
//  isl_ctx *ctx;
//
//  ctx = isl_schedule_node_get_ctx(node);
//  if (!drain_group) {
//    return node;
//  }
//
//  node = isl_schedule_node_child(node, 0);
//  pe_group = polysa_find_pe_group(local_array, drain_group, NULL);
//  tile = polysa_array_ref_group_tile(pe_group);
//
//  access = pe_drain_access(kernel, node, drain_group, read); 
//
//  empty = isl_union_map_is_empty(access);
//  if (empty < 0 || empty) {
//    isl_union_map_free(access);
//    if (empty < 0)
//      return isl_schedule_node_free(node);
//    return polysa_tree_move_up_to_kernel(node);
//  }
//
//  /* Update the group io_dir */
//  if (read) {
//    drain_group->pe_io_dir = (drain_group->pe_io_dir == IO_OUT)? IO_INOUT : IO_IN; 
//  } else {
//    drain_group->pe_io_dir = (drain_group->pe_io_dir == IO_IN)? IO_INOUT : IO_OUT;
//  }
//
//  drain_group->array->global = 1;
//  drain_group->local_array->global = 1;
//
//  /* write.fifoX[D -> A] -> T */
//  from_access = polysa_create_io_access(kernel->ctx, pe_group, drain_group, NULL, read); 
//
//  ma = isl_multi_aff_copy(tile->tiling);
//  ma = isl_multi_aff_pullback_multi_aff(ma,
//      isl_multi_aff_copy(from_access));
//  mpa = isl_multi_pw_aff_from_multi_aff(ma);
//  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);
//
//  domain = isl_union_map_range(access);
//
//  domain = isl_union_set_preimage_multi_aff(domain, from_access);
//  access = isl_union_set_wrapped_domain_map(domain);
//  access = isl_union_map_reverse(access);
//  access = isl_union_map_coalesce(access);
//
//  graft = isl_schedule_node_from_extension(access);
//  graft = isl_schedule_node_child(graft, 0);
//  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);
//
//  while (graft && isl_schedule_node_has_parent(graft))
//    graft = isl_schedule_node_parent(graft);
//
//  if (read) {
//    node = isl_schedule_node_graft_before(node, graft);
//  } else {
//    node = isl_schedule_node_graft_after(node, graft);
//  }
//
//  node = polysa_tree_move_up_to_pe(node);
//
//  return node;
//}

/* The "node" is pointed to the "PE" mark.
 */
static __isl_give isl_schedule_node *add_pe_ext_io_copies(struct polysa_kernel *kernel,
  struct polysa_local_array_info *local_array,
  struct polysa_array_ref_group *io_group,
  __isl_take isl_schedule_node *node, int read)
{
  for (int i = 0; i < io_group->n_ref; i++) {
    struct polysa_stmt_access *ref = io_group->refs[i];
    struct polysa_array_ref_group *pe_group = polysa_find_pe_group(local_array, io_group, ref);
    struct polysa_add_pe_ext_io_copies_data data = {kernel, pe_group, io_group, ref, read, 0, NULL};
    node = isl_schedule_node_map_descendant_bottom_up(node, &add_pe_ext_io_copies_stmt, &data);
  }

  return node;
}

/* The "node" is pointed to the "PE" mark.
 */
static __isl_give isl_schedule_node *add_pe_ext_io_copies_dummy(struct polysa_kernel *kernel,
  struct polysa_local_array_info *local_array,
  struct polysa_array_ref_group *io_group,
  __isl_take isl_schedule_node *node, int read)
{
  isl_union_set *filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  for (int i = 0; i < io_group->n_ref; i++) {
    struct polysa_stmt_access *ref = io_group->refs[i];
    struct polysa_array_ref_group *pe_group = polysa_find_pe_group(local_array, io_group, ref);
    struct polysa_add_pe_ext_io_copies_data data = {kernel, pe_group, io_group, ref, read, 1, NULL};
    node = isl_schedule_node_map_descendant_bottom_up(node, &add_pe_ext_io_copies_stmt, &data);
    filter = isl_union_set_union(filter, data.filter);
  }

  filter = isl_union_set_coalesce(filter);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_filter(node, filter);
  node = isl_schedule_node_parent(node);
  return node;
}

/* Add the statements for copy-in/out the data for array references associated with
 * interior I/O.
 * The "node" is pointed to the "PE" mark.
 */
__isl_give isl_schedule_node *add_pe_int_io_copies(
  struct polysa_kernel *kernel,
  struct polysa_local_array_info *local_array,
  struct polysa_array_ref_group *io_group,
  __isl_take isl_schedule_node *node, int read)
{
  struct polysa_array_tile *tile;
  isl_union_map *access;
  isl_schedule_node *graft;
  int empty;
  isl_multi_aff *from_access;
  isl_multi_aff *ma;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  isl_union_set *domain;
  struct polysa_array_ref_group *pe_group;
  int n_lane = io_group->n_lane;
  isl_printer *p_str;
  char *stmt_name;
 
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  // debug

  node = isl_schedule_node_child(node, 0);
  /* For array references with interior I/O, search for the corresponding PE group. */
  pe_group = polysa_find_pe_group(local_array, io_group, NULL);
  tile = polysa_array_ref_group_tile(pe_group);

  /* Aggregate the copy-in/out access 
   * S -> [D -> A] 
   * S: statement domain elements
   * D: prefix schedule dimensions 
   * A: access */
  access = io_comm_access(kernel, node, io_group, read);
  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return polysa_tree_move_up_to_pe(node);
  }

  /* Update the group io_dir */
  if (read) {
    io_group->pe_io_dir = (io_group->pe_io_dir == IO_OUT)? IO_INOUT: IO_IN;
  } else {
    io_group->pe_io_dir = (io_group->pe_io_dir == IO_IN)? IO_INOUT: IO_OUT;
  }

  pe_group->array->global = 1;
  pe_group->local_array->global = 1;

  /* read.fifoX[D -> A] -> [D -> A] */
  /* Generate statement name */
  p_str = isl_printer_to_str(kernel->ctx);
  if (read)
    p_str = isl_printer_print_str(p_str, "in");
  else
    p_str = isl_printer_print_str(p_str, "out");
  p_str = isl_printer_print_str(p_str, ".");
  if (io_group->group_type != POLYSA_PE_GROUP) {
    p_str = isl_printer_print_str(p_str, "fifo_");
  }
  p_str = isl_printer_print_str(p_str, io_group->array->name);
  if (io_group->group_type == POLYSA_IO_GROUP) {
    if (io_group->local_array->n_io_group > 1) {
      p_str = isl_printer_print_str(p_str, "_");
      p_str = isl_printer_print_int(p_str, io_group->nr);
    }
  } else if (io_group->group_type == POLYSA_DRAIN_GROUP) {
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, "drain");
  }
  p_str = isl_printer_print_str(p_str, ".");
  p_str = isl_printer_print_int(p_str, io_group->n_lane);
  p_str = isl_printer_print_str(p_str, ".1");
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  from_access = polysa_create_io_access_stmt(kernel->ctx, pe_group, io_group, 
      polysa_array_ref_group_tile(pe_group), 
      isl_schedule_node_get_schedule_depth(node), stmt_name);
  free(stmt_name);

  /* [D -> A] -> T */
  ma = isl_multi_aff_copy(tile->tiling);
  ma = isl_multi_aff_pullback_multi_aff(ma,
          isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma);

  /* read.fifoX[D -> A] -> T */
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

  /* [D -> A] */
  domain = isl_union_map_range(access);

//  // debug
//  p = isl_printer_print_union_set(p, domain);
//  printf("\n");
//  // debug

  /* If the array is not a scalar, then we copy in/out the entire
   * tile to/from the local memory. 
   */
  if (read && !polysa_array_is_scalar(io_group->array)) {
    isl_map *map;
    isl_set *set;
    set = isl_map_domain(isl_map_from_union_map(isl_union_set_unwrap(domain)));    
    map = group_tile_buffer(io_group, io_group->pe_tile);
    // TODO: fix it. Inside PEs, set points to the PE schedule dims
    // map domain points to IO shedule dims
//    // debug
//    p = isl_printer_print_map(p, map);
//    printf("\n");
//    p = isl_printer_print_set(p, set);
//    printf("\n");
//    // debug
    map = isl_map_intersect_domain(map, set); 
    domain = isl_union_set_from_set(isl_map_wrap(map));
  }

//  // debug
//  p = isl_printer_print_union_set(p, domain);
//  printf("\n");
//  // debug

  /* read.fifoX[D -> A] */
  domain = isl_union_set_preimage_multi_aff(domain, from_access);
  access = isl_union_set_wrapped_domain_map(domain);
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  if (n_lane > 1) {
    /* Perform data packing */
    int n_index;
    int tile_size[1];
    isl_id *id;
    isl_union_map *umap;
    isl_union_set *filter;

    n_index = isl_schedule_node_band_n_member(graft);
    /* Split off the last dimension */
    graft = isl_schedule_node_band_split(graft, n_index - 1);
    graft = isl_schedule_node_child(graft, 0);
    /* Tile the last dimension */
    tile_size[0] = n_lane;
    graft = polysa_tile_band(graft, tile_size);
    graft = isl_schedule_node_child(graft, 0);
    /* Create a filter */
    filter = schedule_eq_lb(graft);
    graft = isl_schedule_node_insert_filter(graft, filter);
  }

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  if (read) {
    node = isl_schedule_node_graft_before(node, graft);
  } else {
    node = isl_schedule_node_graft_after(node, graft);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  node = polysa_tree_move_up_to_pe(node);

  return node;
}

static void create_pe_module_var(isl_ctx *ctx, struct polysa_array_ref_group *group,
  struct polysa_kernel_var *var, struct polysa_local_array_info *local)
{
  struct polysa_array_tile *tile;
  isl_printer *p;
  isl_val *lcm = isl_val_int_from_si(ctx, 1);

  var->array = group->array;
  var->type = polysa_array_ref_group_type(group);  
  var->n_lane = 1;
  /* Scan all the I/O groups, and compute the lcm of the group SIMD factors,
   * set it as the partition factor of the variable */
  for (int i = 0; i < local->n_io_group; i++) {
    struct polysa_array_ref_group *io_group = local->io_groups[i];
    isl_val *val = isl_val_int_from_si(ctx, io_group->n_lane);
    isl_val *product = isl_val_mul(isl_val_copy(val), isl_val_copy(lcm));
    isl_val *gcd = isl_val_gcd(val, lcm);
    lcm = isl_val_div(product, gcd);
  }
  var->n_part = isl_val_get_num_si(lcm);
  isl_val_free(lcm);

  tile = polysa_array_ref_group_tile(group);

  p = isl_printer_to_str(ctx);
  p = polysa_array_ref_group_print_name(group, p);
  var->name = isl_printer_get_str(p);
  isl_printer_free(p);

  if (tile == NULL) {
    var->size = isl_vec_alloc(ctx, 1);
    var->size = isl_vec_set_element_si(var->size, 0, 1);
  } else {
    var->size = isl_vec_alloc(ctx, group->array->n_index);
    for (int i = 0; i < group->array->n_index; ++i) {
      var->size = isl_vec_set_element_val(var->size, i,
          isl_val_copy(tile->bound[i].size));
    }
  }
}

static void create_io_module_var(isl_ctx *ctx, struct polysa_array_ref_group *group,
  struct polysa_array_tile *tile, struct polysa_kernel_var *var, int n_lane)
{
  isl_printer *p;

  var->array = group->array;
  var->type = polysa_array_ref_group_type(group);
  var->n_lane = n_lane;
  var->n_part = 1;

  p = isl_printer_to_str(ctx);
  p = polysa_array_ref_group_print_name(group, p);
  var->name = isl_printer_get_str(p);
  isl_printer_free(p);

  if (tile == NULL) {
    var->size = isl_vec_alloc(ctx, 1);
    var->size = isl_vec_set_element_si(var->size, 0, 1);
  } else {
    var->size = isl_vec_alloc(ctx, group->array->n_index);
    for (int i = 0; i < group->array->n_index; ++i) {
      isl_val *size;

      size = isl_val_copy(tile->bound[i].size);
      if (n_lane > 1 && i == group->array->n_index - 1) {
        size = isl_val_div(size, isl_val_int_from_si(ctx, n_lane));
      }
      var->size = isl_vec_set_element_val(var->size, i, size);
    }
  }
}

static isl_stat create_pe_module_vars(struct polysa_hw_module *module, struct polysa_kernel *kernel)
{
  int n = 0;
  for (int i = 0; i < kernel->n_array; ++i) {
    struct polysa_local_array_info *array = &kernel->array[i];

    for (int j = 0; j < array->n_pe_group; j++) {    
      struct polysa_array_ref_group *group = array->pe_groups[j];
      enum polysa_group_access_type type;
      
      type = polysa_array_ref_group_type(group);
      if (type != POLYSA_ACCESS_GLOBAL)
        n++;
    }
  }

  module->var = isl_calloc_array(kernel->ctx, struct polysa_kernel_var, n);
  if (!module->var)
    return isl_stat_error;
  module->n_var = n;

  n = 0;
  for (int i = 0; i < kernel->n_array; ++i) {
    struct polysa_local_array_info *array = &kernel->array[i];

    for (int j = 0; j < array->n_pe_group; j++) {
      struct polysa_array_ref_group *group = array->pe_groups[j];
      enum polysa_group_access_type type;
      
      type = polysa_array_ref_group_type(group);
      if (type == POLYSA_ACCESS_GLOBAL)
        continue;
      create_pe_module_var(kernel->ctx, group, &module->var[n], array);
      n++;
    }
  }
 
  return isl_stat_ok;
}

/* We will only create local buffer variables for L2 I/O modules with exterior I/O.
 * For the rest modules, a local register is created later in the codegen.
 */
//static isl_stat create_io_module_vars(struct polysa_hw_module *module, struct polysa_kernel *kernel)
//{
//  int n = 0;
//  if (module->type == IO_MODULE && module->level == 2) {
//    if (module->io_groups[0]->io_type == POLYSA_EXT_IO) {
//      n++;
//    }
//  }
//  if (module->type == DRAIN_MODULE && module->level == 1) {
//    n++;
//  }
//
//  module->var = isl_calloc_array(kernel->ctx, struct polysa_kernel_var, n);
//  if (!module->var)
//    return isl_stat_error;
//  module->n_var = n;
//
//  if (n > 0) 
//    create_io_module_var(kernel->ctx, module->io_groups[0], &module->var[0]);
//
//  return isl_stat_ok;
//}

static isl_stat create_io_module_vars(struct polysa_hw_module *module, struct polysa_kernel *kernel, struct polysa_array_tile *tile)
{
  module->var = isl_calloc_array(kernel->ctx, struct polysa_kernel_var, 1);
  if (!module->var)
    return isl_stat_error;
  module->n_var = 1;

  create_io_module_var(kernel->ctx, module->io_groups[0], tile, &module->var[0], module->data_pack_inter);

  return isl_stat_ok;
}

static __isl_give isl_schedule *pe_module_dummy_gen(struct polysa_gen *gen,
  struct polysa_hw_module *module, struct polysa_array_ref_group *group)
{
  isl_schedule *schedule;
  isl_schedule_node *node;
  isl_id *id, *hw_id;
  struct polysa_kernel *kernel;

  schedule = gen->schedule;
  schedule = isl_schedule_dup(schedule);
  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);
  node = polysa_tree_move_down_to_kernel(node);

  id = isl_schedule_node_mark_get_id(node);
  kernel = (struct polysa_kernel *)isl_id_get_user(id);
  isl_id_free(id);

  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  node = split_band(node, kernel->n_sa_dim);

  node = polysa_tree_move_down_to_pe(node, kernel->core);

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  node = add_pe_ext_io_copies_dummy(kernel, group->local_array, group, node, 1);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Insert "pipeline" mark under the last "latency" mark */
  node = isl_schedule_node_map_descendant_bottom_up(node,
      &insert_pipeline_mark, kernel);

  /* Insert "unroll" mark under the last "simd" mark */
  node = isl_schedule_node_map_descendant_bottom_up(node,
      &insert_unroll_mark, kernel);

  /* Add module mark after the kernel mark */
  hw_id = isl_id_alloc(gen->ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, hw_id);

  /* Add the PE id filter */
  node = polysa_tree_move_up_to_kernel(node);
  isl_schedule_node_child(node, 0);
  node = insert_context(kernel, node);
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_filter(node,
      isl_union_set_copy(kernel->pe_filter));

  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  return schedule;
}

/* Modify the input "schedule" to describe the PE module.
 * Set the schedule dimensions of space loops as parameters.
 *
 * For interior I/O groups
 * - add copy-in before PE computation (RAW, RAR)
 * - add copy-out after PE computation (RAW)
 *   - domain: S -> type[D -> access]
 *   - schedule: type[D -> access] -> tiling
 * For exterior I/O groups
 *   for each access in the group
 *   - add copy-in before user statement (RAW, RAR)
 *   - add copy-out after user statement (RAW, RAR)
 *     - domain: S -> type[D -> access]
 *     - schedule: type[D -> access] -> tiling (if any, otherwise, create a register tiling)
 * For WAW group 
 * - for each access in the group
 *   - add write-out after user statement (WAW)
 *     - domain: S -> type[D -> access]
 *     - schedule: type[D -> access] -> tiling
 */
__isl_give struct polysa_hw_module *sa_pe_module_gen(struct polysa_gen *gen)
{
  isl_schedule_node *node;
  isl_id *id;
  struct polysa_kernel *kernel;
  isl_schedule *schedule, *new_schedule;
  int single_statement;
  isl_union_set *domain;
  struct polysa_hw_module *module;
  isl_id *hw_id;

//  // debug
//  isl_printer *p = isl_printer_to_file(gen->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  module = polysa_hw_module_alloc();

  /* Add the filters for PEs */
  schedule = gen->schedule;
  schedule = isl_schedule_dup(schedule);
  node = isl_schedule_get_root(schedule);
  node = polysa_tree_move_down_to_kernel(node);
  
  id = isl_schedule_node_mark_get_id(node);
  kernel = (struct polysa_kernel *)isl_id_get_user(id);
  isl_id_free(id);
  single_statement = kernel->single_statement;
  domain = isl_schedule_node_get_domain(node);

  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  node = split_band(node, kernel->n_sa_dim);
  kernel->pe_ids = ppcg_scop_generate_names(gen->prog->scop,
      kernel->n_sa_dim, "p");
  kernel->pe_filter = set_schedule_modulo(node, kernel->pe_ids,
      kernel->sa_dim);
  kernel->sa_grid_size = extract_sa_grid_size(kernel, domain);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Add the statements for I/O groups with exterior I/O at the user 
   * statement level. 
   * Add the statements for I/O group with interior I/O at the PE level.
   */
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  /* Add copy-in/copy-out statements */
  for (int i = 0; i < kernel->n_array; ++i) {
    struct polysa_local_array_info *array = &kernel->array[i];
    for (int j = 0; j < array->n_io_group; j++) {
      struct polysa_array_ref_group *group = array->io_groups[j];
      if (group->array_io_dir == IO_NULL)
        continue;
      if (group->local_array->array_type == POLYSA_EXT_ARRAY) {
        node = add_pe_ext_io_copies(kernel, array, group, node, 0);
        node = add_pe_ext_io_copies(kernel, array, group, node, 1);
      } else if (group->local_array->array_type == POLYSA_INT_ARRAY) {
        if (group->io_type == POLYSA_INT_IO) {
          node = add_pe_int_io_copies(kernel, array, group, node, 0);  
          node = add_pe_int_io_copies(kernel, array, group, node, 1); 
        } else {
          node = add_pe_ext_io_copies(kernel, array, group, node, 0); 
          node = add_pe_ext_io_copies(kernel, array, group, node, 1); 
        }
      }
      
      module->n_io_group++;
      module->io_groups = (struct polysa_array_ref_group **)realloc(module->io_groups,
        module->n_io_group * sizeof(struct polysa_array_ref_group *));
      module->io_groups[module->n_io_group - 1] = group;
    }
    if (array->drain_group && array->drain_group->array_io_dir != IO_NULL) {
      node = add_pe_ext_io_copies(kernel, array, array->drain_group, node, 0);

      module->n_io_group++;
      module->io_groups = (struct polysa_array_ref_group **)realloc(module->io_groups,
        module->n_io_group * sizeof(struct polysa_array_ref_group *));
      module->io_groups[module->n_io_group - 1] = array->drain_group;
    }
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Insert "pipeline" mark under the last "latency" mark */
  node = isl_schedule_node_map_descendant_bottom_up(node,
      &insert_pipeline_mark, kernel);

  /* Insert "unroll" mark under the last "simd" mark */
  node = isl_schedule_node_map_descendant_bottom_up(node,
      &insert_unroll_mark, kernel);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Add module mark after the kernel mark */
  hw_id = isl_id_alloc(gen->ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, hw_id);

  /* Add the PE id filter */
  node = polysa_tree_move_up_to_kernel(node);
  isl_schedule_node_child(node, 0);
  node = insert_context(kernel, node); 
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_filter(node, 
      isl_union_set_copy(kernel->pe_filter));

  isl_schedule_free(schedule);
  new_schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

//  // debug
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, new_schedule);
//  printf("\n");
//  // debug

  module->sched = new_schedule;
  module->type = PE_MODULE;
  module->name = strdup("PE");
  module->inst_ids = isl_id_list_copy(kernel->pe_ids);
  create_pe_module_vars(module, kernel);
  module->kernel = kernel;

  /* Add the dummy module */
  module->n_pe_dummy_modules = 0;
  module->pe_dummy_modules = NULL;
  for (int i = 0; i < kernel->n_array; ++i) {
    struct polysa_local_array_info *array = &kernel->array[i];
    if (array->array_type == POLYSA_INT_ARRAY)
      continue;
    for (int j = 0; j < array->n_io_group; j++) {
      struct polysa_array_ref_group *group = array->io_groups[j];
      if (group->pe_io_dir != IO_INOUT)
        continue;
      /* Generate the dummy module */
      isl_schedule *sched;
      sched = pe_module_dummy_gen(gen, module, group);
//      // TODO
//      // debug
//      p = isl_printer_print_schedule(p, sched);
//      p = isl_printer_flush(p);
//      // debug
      module->n_pe_dummy_modules++;
      module->pe_dummy_modules = 
        (struct polysa_pe_dummy_module **)realloc(module->pe_dummy_modules,
            module->n_pe_dummy_modules * sizeof(struct polysa_pe_dummy_module *));
      struct polysa_pe_dummy_module *dummy_module = polysa_pe_dummy_module_alloc();
      dummy_module->module = module;
      dummy_module->io_group = group;
      dummy_module->sched = sched;
      module->pe_dummy_modules[module->n_pe_dummy_modules - 1] = dummy_module;
    }
  }

  return module;
}

/* Generate two prefixes: fifo_prefix and buffer_prefix
 * fifo_prefix: fifo_A_0
 * buffer_prefix: local_A_0
 */
static void init_suffix(struct polysa_hw_module *module, 
  struct polysa_array_ref_group *group, char **fifo_suffix, char **buf_suffix) 
{
  isl_ctx *ctx = isl_map_get_ctx(group->access);

  isl_printer *p = isl_printer_to_str(ctx);
  p = polysa_array_ref_group_print_fifo_name(group, p);
  *fifo_suffix = isl_printer_get_str(p);
  isl_printer_free(p);

  p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, "local_");
  p = isl_printer_print_str(p, group->array->name);
  if ((group->group_type == POLYSA_IO_GROUP && group->local_array->n_io_group > 1) ||
    (group->group_type == POLYSA_PE_GROUP && group->local_array->n_pe_group > 1))
  {
    p = isl_printer_print_str(p, "_");  
    p = isl_printer_print_int(p, group->nr);
  }  
  if (group->group_type == POLYSA_DRAIN_GROUP) {
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_str(p, "drain");
  }
  *buf_suffix = isl_printer_get_str(p);
  isl_printer_free(p);
}

struct add_io_copies_stmt_acc_data {
  struct polysa_kernel *kernel;
  struct polysa_array_ref_group *group;
  struct polysa_stmt_access *ref;
  struct polysa_array_tile *local_tile;
  int n_lane;
  int read;
  char *stmt_name;
};

static __isl_give isl_schedule_node *add_io_copies_stmt_acc_single(__isl_take isl_schedule_node *node, void *user)
{
  struct add_io_copies_stmt_acc_data *data = (struct add_io_copies_stmt_acc_data *)(user);
  struct polysa_array_ref_group *group = data->group;
  struct polysa_stmt_access *ref = data->ref;
  char *stmt_name = data->stmt_name;
  int read = data->read;
  isl_union_set *uset, *empty_filter, *domain;
  isl_set *set;
  isl_space *space;
  isl_id *id, *id2;
  isl_ctx *ctx;
  isl_union_map *access;
  int empty;
  struct polysa_array_tile *tile;
  isl_multi_aff *ma, *from_access;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  isl_schedule_node *graft;
  int n_lane = data->n_lane;
  int is_simd;
  isl_id *hls_id;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  uset = isl_schedule_node_get_domain(node);
  set = isl_set_from_union_set(isl_union_set_copy(uset));
  space = isl_set_get_space(set);
  isl_set_free(set);  
  id = isl_space_get_tuple_id(space, isl_dim_set);
  isl_space_free(space);
  space = isl_map_get_space(ref->access);
  id2 = isl_space_get_tuple_id(space, isl_dim_in);
  empty_filter = isl_union_set_empty(isl_union_set_get_space(uset));
  isl_union_set_free(uset);
  isl_space_free(space);
//  // debug
//  p = isl_printer_print_id(p, id);
//  p = isl_printer_flush(p);
//  p = isl_printer_print_id(p, id2);
//  p = isl_printer_flush(p);
//  // debug
  if (id != id2) {
    isl_id_free(id);
    isl_id_free(id2);
    node = isl_schedule_node_insert_filter(node, empty_filter);
    return node;
  }
  isl_id_free(id);
  isl_id_free(id2);
  ctx = isl_schedule_node_get_ctx(node);
  is_simd = is_node_under_simd(node);

  access = io_comm_access_ref(data->kernel, node, group, ref, read);
  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    isl_union_set_free(empty_filter);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return node;
  }

  from_access = polysa_create_io_access_stmt(ctx, group, group, data->local_tile, isl_schedule_node_get_schedule_depth(node), stmt_name);
  free(stmt_name);

  /* Create a register tiling */
  tile = create_register_tiling(node, group, ref);
  ma = isl_multi_aff_copy(tile->tiling);
  ma = isl_multi_aff_pullback_multi_aff(ma, 
      isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma);
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

  domain = isl_union_map_range(access);
  if (read && !polysa_array_is_scalar(group->array)) {
    isl_map *map;
    isl_set *set;
    set = isl_map_domain(isl_map_from_union_map(isl_union_set_unwrap(domain)));
    map = group_tile_buffer(group, tile); 
    map = isl_map_intersect_domain(map, set);
    domain = isl_union_set_from_set(isl_map_wrap(map));
  }

  domain = isl_union_set_preimage_multi_aff(domain, from_access);
  access = isl_union_set_wrapped_domain_map(domain);
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);
  
  if (n_lane > 1 && is_simd) {
    /* The loop above is the SIMD loop */
    int n_index;
    int tile_size[1];
    isl_id *id;
    isl_printer *p_str;
    isl_union_map *umap;
    isl_union_set *filter;

    /* Create a filter */
    node = isl_schedule_node_parent(node);
    filter = schedule_eq_lb(node);
    node = isl_schedule_node_insert_filter(node, filter);
    node = isl_schedule_node_child(node, 0);
    node = isl_schedule_node_child(node, 0);
  }

  /* Insert a "pipeline" mark under the band node */
  hls_id = isl_id_alloc(ctx, "hls_pipeline", NULL);
//  graft = isl_schedule_node_insert_mark(graft, hls_id);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_mark(graft, hls_id);
  graft = isl_schedule_node_parent(graft);

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  node = isl_schedule_node_graft_before(node, graft);
  node = isl_schedule_node_insert_filter(node, empty_filter);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  node = isl_schedule_node_parent(node);
  node = isl_schedule_node_parent(node);
  node = isl_schedule_node_parent(node);

  polysa_array_tile_free(tile);

  return node;
}

/* Add copies at the stmt level for each array reference in the "group" in the I/O modules.
 * "group" is an I/O group.
 * "read" denotes if copy-in or copy-out from/to the external memory.
 * "in" denotes the fifo direction.
 */
__isl_give isl_schedule_node *add_io_copies_stmt_acc(
  struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group,
  __isl_take isl_schedule_node *node,
  struct polysa_array_tile *tile,
  int n_lane,
  int read,
  __isl_take char *stmt_name,
  int before
) {
  struct add_io_copies_stmt_acc_data data = {kernel, group, NULL, tile, n_lane, read, stmt_name};

  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    data.ref = ref;
    node = isl_schedule_node_map_descendant_bottom_up(node, &add_io_copies_stmt_acc_single, &data);
  }

  return node;
}

/* If "is_buffer" is set, add a marker for dependence false. This is
 * only for Xilinx platform.
 */
static __isl_give isl_schedule_node *add_io_copies_stmt_tile(
  struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group,
  __isl_take isl_schedule_node *node,
  struct polysa_array_tile *local_tile, /* Local buffer */
  struct polysa_array_tile *tile, /* The tile to be copied */
  int n_lane,
  int read,
  __isl_take char *stmt_name,
  int before, int is_buffer)
{
  isl_union_map *access = NULL;
  int empty;
  isl_multi_aff *from_access;
  isl_multi_aff *ma;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  isl_union_set *domain;
  isl_schedule_node *graft;
  int n;
  isl_id *id;
  isl_ctx *ctx = kernel->ctx;

//  // debug
//  isl_printer *p = isl_printer_to_file(kernel->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  // debug

  access = io_comm_access(kernel, node, group, read);

  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return node;
  }

  from_access = polysa_create_io_access_stmt(kernel->ctx, group, group, local_tile, isl_schedule_node_get_schedule_depth(node), stmt_name);

  ma = isl_multi_aff_copy(tile->tiling);

//  // debug
//  p = isl_printer_print_multi_aff(p, ma);
//  printf("\n");
//  p = isl_printer_print_multi_aff(p, from_access);
//  printf("\n");
//  // debug

  ma = isl_multi_aff_pullback_multi_aff(ma, 
      isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma);
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

  domain = isl_union_map_range(access);
  if (read && !polysa_array_is_scalar(group->array)) {
    isl_map *map;
    isl_set *set;
    set = isl_map_domain(isl_map_from_union_map(isl_union_set_unwrap(domain)));
    map = group_tile_buffer(group, tile); 
    map = isl_map_intersect_domain(map, set);
    domain = isl_union_set_from_set(isl_map_wrap(map));
  }

  domain = isl_union_set_preimage_multi_aff(domain, from_access);
  access = isl_union_set_wrapped_domain_map(domain);
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

//  // debug
//  p = isl_printer_print_schedule_node(p, graft);
//  p = isl_printer_flush(p);
//  // debug

  /* Split off the last dimension */
  n = isl_schedule_node_band_n_member(graft);
  if (n > 1) {
    graft = isl_schedule_node_band_split(graft, n - 1);
    graft = isl_schedule_node_child(graft, 0);
  }

  /* Insert a coalesce mark */
  id = isl_id_alloc(ctx, "access_coalesce", NULL);
  graft = isl_schedule_node_insert_mark(graft, id);
  graft = isl_schedule_node_child(graft, 0);

  if (n_lane > 1) {
    /* Peform data packing */
    int tile_size[1];
    isl_id *id;
    isl_printer *p_str;
    isl_union_map *umap;
    isl_union_set *filter;

    /* Tile the last dimension */
    tile_size[0] = n_lane;
    graft = polysa_tile_band(graft, tile_size);
    graft = isl_schedule_node_child(graft, 0);
    /* Create a filter */
    filter = schedule_eq_lb(graft);
    graft = isl_schedule_node_insert_filter(graft, filter);

    /* Move to the tile loop */
    graft = isl_schedule_node_parent(graft);
  }
  free(stmt_name);
  /* Insert a "pipeline" mark inside the band node */
  id = isl_id_alloc(ctx, "hls_pipeline", NULL);
//  graft = isl_schedule_node_insert_mark(graft, id);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_mark(graft, id);
  graft = isl_schedule_node_parent(graft);

  if (is_buffer && !read) {
    /* Insert a "dependence" mark */
    char *mark_name;
    isl_printer *p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, "hls_dependence.");
    p_str = polysa_array_ref_group_print_name(group, p_str);
    mark_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    id = isl_id_alloc(ctx, mark_name, NULL);
    graft = isl_schedule_node_insert_mark(graft, id);
    free(mark_name);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, graft);
//  p = isl_printer_flush(p);
//  // debug

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  if (before) {
    node = isl_schedule_node_graft_before(node, graft);
  } else {
    node = isl_schedule_node_graft_after(node, graft);
  }

  return node;
   
}

/* Add copies at the "node" level with the array references in the "group" in the I/O modules.
 * "group" is I/O group.
 * "read" denotes if copy-in or copy-out from/to the external memory.
 * "in" denotes the fifo direction.
 */
__isl_give isl_schedule_node *add_io_copies_suffix(
  struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group,
  __isl_take isl_schedule_node *node,
  int read,
  int in,
  __isl_take char *suffix,
  int before)
{
  struct polysa_array_tile *tile;
  isl_union_map *access = NULL;
  int empty;
  isl_multi_aff *from_access;
  isl_multi_aff *ma;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  isl_union_set *domain;
  isl_schedule_node *graft;

  tile = polysa_array_ref_group_tile(group);

  if (group->group_type == POLYSA_IO_GROUP) {
    if (group->io_type == POLYSA_INT_IO)
      access = pe_int_comm_access(kernel, node, group, read);
    else {
      for (int i = 0; i < group->n_ref; i++) {
        struct polysa_stmt_access *ref_i = group->refs[i];
        if (!access)
          access = pe_ext_comm_access(kernel, node, group, ref_i, read); 
        else
          access = isl_union_map_union(access, pe_ext_comm_access(kernel, node, group, ref_i, read));
      }
    }
  } else if (group->group_type == POLYSA_DRAIN_GROUP) {
    access = pe_drain_access(kernel, node, group, read);
  }

  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    free(suffix);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return node;
  }

  from_access = polysa_create_io_access_suffix(kernel->ctx, group, group, tile, in, 1, suffix);
  free(suffix);

  ma = isl_multi_aff_copy(tile->tiling);
  ma = isl_multi_aff_pullback_multi_aff(ma, 
      isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma);
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

  domain = isl_union_map_range(access);
  if (read && !polysa_array_is_scalar(group->array)) {
    isl_map *map;
    isl_set *set;
    set = isl_map_domain(isl_map_from_union_map(isl_union_set_unwrap(domain)));
//    isl_union_set_free(domain);
    map = group_tile(group); 
    map = isl_map_intersect_domain(map, set);
    domain = isl_union_set_from_set(isl_map_wrap(map));
  }

  domain = isl_union_set_preimage_multi_aff(domain, from_access);
  access = isl_union_set_wrapped_domain_map(domain);
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  if (before) {
    node = isl_schedule_node_graft_before(node, graft);
  } else {
    node = isl_schedule_node_graft_after(node, graft);
  }

  return node;
}

struct polysa_add_L2_ext_io_copies_data {
  struct polysa_kernel *kernel;
  struct polysa_hw_module *module;
  struct polysa_array_ref_group *group;
  struct polysa_stmt_access *ref;
  int read;
  isl_union_set *filter;
};

static __isl_give isl_schedule_node *add_ext_io_L2_copies_stmt(__isl_take isl_schedule_node *node, void *user)
{
  struct polysa_add_L2_ext_io_copies_data *data = (struct polysa_add_L2_ext_io_copies_data *)(user);
  struct polysa_array_ref_group *group = data->group;
  struct polysa_hw_module *module = data->module;
  struct polysa_array_tile *tile;
  struct polysa_stmt_access *ref = data->ref;
  int read = data->read;
  isl_union_set *domain;
  isl_union_set *user_domain;
  isl_space *space, *acc_space;
  isl_id *id;
  isl_ctx *ctx;
  int empty;
  isl_multi_aff *from_access;
  isl_multi_aff *ma;
  isl_multi_pw_aff *mpa;
  isl_multi_union_pw_aff *mupa;
  isl_schedule_node *graft;
  isl_union_map *access;
  isl_id *ref_id;
  isl_union_map *sched;
  isl_map *acc;
  isl_bool ok;
  isl_union_map *ref_access;
  char *fifo_suffix, *buf_suffix;
  isl_union_set *filter;
  char *suffix;
  isl_set *set;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  user_domain = isl_schedule_node_get_domain(node); 
  set = isl_set_from_union_set(isl_union_set_copy(user_domain));
  space = isl_set_get_space(set);
  isl_set_free(set);
  id = isl_space_get_tuple_id(space, isl_dim_set); 
  acc_space = isl_map_get_space(ref->access); 
  ref_id = isl_space_get_tuple_id(acc_space, isl_dim_in); 
  filter = isl_union_set_empty(isl_union_set_get_space(user_domain)); 
  isl_union_set_free(user_domain);
  isl_space_free(acc_space);
  isl_space_free(space); 
  if (id != ref_id) {
    isl_id_free(id);
    isl_id_free(ref_id);
    node = isl_schedule_node_insert_filter(node, filter);
    return node;
  }

  isl_id_free(id);
  isl_id_free(ref_id);

  ctx = isl_schedule_node_get_ctx(node);
  access = pe_ext_comm_access(data->kernel, node, group, ref, read);
  empty = isl_union_map_is_empty(access);
  if (empty < 0 || empty) {
    isl_union_map_free(access);
    isl_union_set_free(filter);
    if (empty < 0)
      return isl_schedule_node_free(node);
    return polysa_tree_move_up_to_pe(node);
  }

  init_suffix(module, group, &fifo_suffix, &buf_suffix);

  /* Create a register tiling. */
  tile = create_register_tiling(node, group, ref);
  suffix = concat(ctx, fifo_suffix, "local");
  from_access = polysa_create_io_access_suffix(ctx, group, group, tile, read, 0, suffix);
  free(suffix);
  ma = isl_multi_aff_copy(tile->tiling);
  ma = isl_multi_aff_pullback_multi_aff(ma, isl_multi_aff_copy(from_access));
  mpa = isl_multi_pw_aff_from_multi_aff(ma);
  mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);
  domain = isl_union_map_range(access);
  domain = isl_union_set_preimage_multi_aff(domain, from_access);
  access  = isl_union_set_wrapped_domain_map(domain);
  access = isl_union_map_reverse(access);
  access = isl_union_map_coalesce(access);

  graft = isl_schedule_node_from_extension(access);
  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  if (read) {
    node = isl_schedule_node_graft_before(node, graft);
  } else {
    node = isl_schedule_node_graft_after(node, graft);
  }

  node = isl_schedule_node_insert_filter(node, filter);

  
  node = isl_schedule_node_parent(node);
  node = isl_schedule_node_parent(node);
  node = isl_schedule_node_parent(node);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  polysa_array_tile_free(tile);
  free(fifo_suffix);
  free(buf_suffix);

  return node;
}

/* "node" points the first node below the pe mark. 
 * Add the copy-in/out statments at the user stmt level.
 */
static __isl_give isl_schedule_node *add_ext_io_L2_copies_stmt_wrap(
  struct polysa_kernel *kernel, 
  struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, 
  __isl_take isl_schedule_node *node, int read)
{
  struct polysa_add_L2_ext_io_copies_data data = {kernel, module, group, NULL, read, NULL};

  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref_i = group->refs[i];
    data.ref = ref_i;
    node = isl_schedule_node_map_descendant_bottom_up(node, &add_ext_io_L2_copies_stmt, &data);
  }

  return node;
}

/* The "node" points to the io_L2 mark.
 * For exterior I/O:
 * If read:
 * at io_L1
 * - add D -> in_filter.fifoX.cX.io_L2_u0[D -> A] | 
 *       trans_filter.fifoX.cX.io_L2_u0[D -> A] -> T[group]
 * at user stmt
 * - add D -> out.fifoX_local[D -> A] | 
 *       out.fifoX_local[D -> A] -> T[group]
 *
 * If write:
 * at user stmt
 * - add D -> in.fifoX_local[D -> A] | 
 *       in.fifoX_local[D -> A] -> T[group]
 * at io_L1
 * - add D -> out_filter.fifoX.cX.io_L2_u0[D -> A] | 
 *       trans_filter.fifoX.cX.io_L2_u0[D -> A] -> T[group]
  */
//__isl_give isl_schedule_node *add_ext_io_L2_copies(struct polysa_kernel *kernel,
//  struct polysa_hw_module *module,
//  struct polysa_array_ref_group *group,
//  __isl_take isl_schedule_node *node,
//  int read)
//{
//  char *fifo_suffix, *buf_suffix;
//  isl_ctx *ctx;
//  isl_union_set *eq_filter, *neq_filter, *filter;
//  isl_union_set *domain;
//  isl_id_list *io_ids;
//  isl_union_set_list *filters;
//  isl_bool insert_eq = isl_bool_false;
//  isl_bool insert_neq = isl_bool_false;
//  int depth;
//  isl_printer *p_str;
//  int is_filter;
//  char *stmt_name;
// 
//  init_suffix(module, group, &fifo_suffix, &buf_suffix);
//  ctx = isl_schedule_node_get_ctx(node);
//
//  node = isl_schedule_node_child(node, 0);
//  depth = isl_schedule_node_get_schedule_depth(node);
//  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
//    is_filter = 1;
//  } else {
//    is_filter = 0;
//  }
//  node = polysa_tree_move_down_to_mark(node, kernel->core, "io_L1");
//  p_str = isl_printer_to_str(ctx);
//  if (read) {
//    p_str = isl_printer_print_str(p_str, "in_trans");
//  } else {
//    p_str = isl_printer_print_str(p_str, "out_trans");
//  }
//  if (is_filter)
//    p_str = isl_printer_print_str(p_str, "_filter_buf");
//  p_str = isl_printer_print_str(p_str, ".");
//  p_str = isl_printer_print_str(p_str, fifo_suffix);
//  p_str = isl_printer_print_str(p_str, ".");
//  if (is_filter) {
//    p_str = isl_printer_print_int(p_str, depth);
//    p_str = isl_printer_print_str(p_str, ".");
//    p_str = isl_printer_print_int(p_str, 0);
//  }
//  stmt_name = isl_printer_get_str(p_str);
//  isl_printer_free(p_str);
//
//  node = add_io_copies_stmt(kernel, group, node, read, read, stmt_name, read ? 1 : 0);
//
//  node = polysa_tree_move_down_to_pe(node, kernel->core);
//  node = isl_schedule_node_child(node, 0);
//  /* Add the user stmt extension nodes. */
//  node = add_ext_io_L2_copies_stmt_wrap(kernel, module, group, node, !read); 
//
//  node = polysa_tree_move_up_to_mark(node, "io_L2");
//
//  free(fifo_suffix);
//  free(buf_suffix);
//
//  return node;
//}

/* Insert the statements for copy-in/out data with interior I/O inside I/O modules.
 * The node is pointed at the leaf node under the PE mark.
 * For interior I/O:
 * If read:
 * - add D -> fifoX_in.read[D -> A] | fifoX_in.read[D -> A] -> T[group]
 * - add D' -> fifoX_local_out.write[D' -> A] | fifoX_local_out.write[D' -> A] -> T[group]
 *   D' = D with cX == level
 * - add D'' -> fifoX_out.write[D'' -> A] | fifoX_out.write[D'' -> A] -> T[group]
 *   D'' = D with cX != level
 *
 * If write:
 * - add D' -> fifoX_local_in.read[D' -> A] | fifoX_local_in.read[D' -> A] -> T[group]
 *   D' = D with cX == level
 * - add D'' -> fifoX_in.read[D'' -> A] | fifoX_in.read[D'' -> A] -> T[group]
 *   D'' = D with cX != level
 * - add D -> fifoX_out.write[D -> A] | fifoX_out.write[D -> A] -> T[group]
 */
__isl_give isl_schedule_node *add_int_io_copies(
  struct polysa_kernel *kernel,
  struct polysa_hw_module *module,
  struct polysa_array_ref_group *group,
  __isl_take isl_schedule_node *node,
  int read, 
  __isl_keep char *level)
{
  char *fifo_suffix, *buf_suffix;
  isl_union_set *eq_filter, *neq_filter;
  isl_id_list *io_ids;
  isl_bool insert_filter = isl_bool_false;
  isl_union_set *domain;
  isl_union_set_list *filters;

  return node;

  init_suffix(module, group, &fifo_suffix, &buf_suffix);
  isl_ctx *ctx = isl_schedule_node_get_ctx(node);

  /* Create filters */
  node = polysa_tree_move_up_to_mark(node, level);
  node = isl_schedule_node_child(node, 0);
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    char *str = concat(ctx, level, "u") ;
    io_ids = ppcg_scop_generate_names(kernel->scop, 1, str);
    eq_filter = set_schedule_eq(node, io_ids);
    neq_filter = set_schedule_neq(node, io_ids);
    insert_filter = isl_bool_true;
    free(str);
    isl_id_list_free(io_ids);
  }

  node = polysa_tree_move_down_to_pe(node, kernel->core); 
  node = isl_schedule_node_child(node, 0);
  if (insert_filter) {
    filters = isl_union_set_list_from_union_set(eq_filter);
    filters = isl_union_set_list_insert(filters, 1, neq_filter);
  }

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Insert statement fifoX_in.read/fifoX_out.write. */
  node = add_io_copies_suffix(kernel, group, node, read, read, 
      strdup(fifo_suffix), read? 1 : 0);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Insert statement with filters. */
  if (insert_filter) {
//    isl_schedule_node *seq_node;

    node = isl_schedule_node_insert_sequence(node, filters);
//    seq_node = isl_schedule_node_copy(node);

//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
    node = isl_schedule_node_child(node, 0);
    node = isl_schedule_node_child(node, 0);
    node = add_io_copies_suffix(kernel, group, node, read, !read, 
        concat(ctx, fifo_suffix, "local"), 1);

//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug

    node = isl_schedule_node_cut(node);
//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
//    isl_schedule_node_free(node);
    node = isl_schedule_node_parent(node); // filter
    node = isl_schedule_node_parent(node); // sequence
    node = isl_schedule_node_parent(node); // extension
    node = isl_schedule_node_parent(node); // filter
    node = isl_schedule_node_parent(node); // sequence

//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug

    node = isl_schedule_node_child(node, 1);
    node = isl_schedule_node_child(node, 0);
    node = add_io_copies_suffix(kernel, group, node, read, !read, strdup(fifo_suffix), 1);
    node = isl_schedule_node_cut(node);
  } else {
    node = add_io_copies_suffix(kernel, group, node, read, !read, 
        concat(ctx, fifo_suffix, "local"), 1);
    node = isl_schedule_node_cut(node);
  }
  
  node = polysa_tree_move_up_to_pe(node);

  free(fifo_suffix);
  free(buf_suffix);

  return node;
}

/* "node" points to kernel "mark". 
 * We first construct a relation "local"
 *
 *  [[D -> R] -> [D' -> R']]
 *
 * of pairs of domain iterations accessing the reference group and 
 * references in the group that are coscheduled by "sched".
 *
 * We will remove the access in the group that intersect with this relation.
 * If there is still any access left, then the copy-in and copy-out sets 
 * are not fuilly overlapped, and we will return false. Otherwise, we return true.
 */
isl_bool internal_group_in_out_overlap(
  __isl_keep isl_schedule_node *node,
  struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, int read)
{
  int empty;
  struct polysa_prog *prog = kernel->prog;
  isl_union_pw_multi_aff *tagger;
  isl_union_map *prefix;
  isl_union_map *access, *tagged;
  isl_union_set *domain;
  isl_set *prefix_range;
  isl_map *lt;
  int n_sched_dim;
  isl_union_map *overlap;
  isl_union_map *external, *universe;
  isl_union_set *access_domain;
  isl_union_set *tag_set;
  isl_map *sched_identity;
  int pe_depth, array_depth;

  node = isl_schedule_node_copy(node);
  node = polysa_tree_move_down_to_array(node, kernel->core);
  array_depth = isl_schedule_node_get_schedule_depth(node);
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  pe_depth = isl_schedule_node_get_schedule_depth(node);
  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
              isl_union_pw_multi_aff_copy(kernel->contraction)); 
  isl_schedule_node_free(node);
  access = polysa_io_group_access_relation(group, read, !read); 
  tagged = group_tagged_access_relation(group); 
 
  /* Remove the local dependency first */
  access = remove_local_accesses_group_flow(kernel, group, access, prefix, read);

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_map_get_ctx(tagged), stdout);
//  // debug

  /* Tagger maps the tagged iteration domain to untagged iteration domain.
   * Iteration domain is tagged to the access function.
   * e.g. [S1[i,j,k] -> _pet_ref_1[]] -> S1[(i),(j),(k)]
   */
  tagger = isl_union_pw_multi_aff_copy(prog->scop->tagger);
  domain = isl_union_map_domain(isl_union_map_copy(tagged));
  tagger = isl_union_pw_multi_aff_intersect_domain(tagger, 
            isl_union_set_copy(domain));
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix, tagger); 
  
  prefix_range = isl_set_from_union_set(isl_union_map_range(isl_union_map_copy(prefix))); 
  n_sched_dim = isl_set_dim(prefix_range, isl_dim_set);
//  // debug
//  p = isl_printer_print_set(p, prefix_range);
//  printf("\n");
//  // debug

  sched_identity = isl_set_identity(isl_set_copy(prefix_range));
  lt = isl_map_lex_lt_first(isl_map_get_space(sched_identity), array_depth); 
//  isl_set_free(prefix_range);
  isl_map_free(sched_identity);

//  // debug
//  p = isl_printer_print_map(p, lt);
//  printf("\n");
//  // debug

  /* Set the space dims equal. */
  for (int i = array_depth; i < n_sched_dim; i++) {
    lt = isl_map_equate(lt, isl_dim_in, i, isl_dim_out, i);
  }
  lt = isl_map_intersect_domain(lt, isl_set_copy(prefix_range));
  lt = isl_map_intersect_range(lt, prefix_range);

//  // debug
//  p = isl_printer_print_map(p, lt);
//  printf("\n");
//  // debug
  lt = isl_map_lexmin(lt); 
  
  overlap = isl_union_map_apply_range(isl_union_map_copy(prefix), isl_union_map_from_map(lt));
  overlap = isl_union_map_apply_range(overlap, isl_union_map_reverse(prefix)); 

  /* Derive the overlapping set. */
  overlap = isl_union_map_intersect(overlap, 
              isl_union_map_copy(prog->scop->tagged_dep_flow)); 
  empty = isl_union_map_is_empty(overlap);

  external = isl_union_map_copy(prog->scop->tagged_dep_flow); 
  universe = isl_union_map_universe(isl_union_map_copy(access)); 
  access_domain = isl_union_map_domain(universe); 
  domain = isl_union_set_universe(domain); 
  universe = isl_union_set_unwrap(domain); 
  universe = isl_union_map_intersect_domain(universe, access_domain); 
  /* D -> __pet_ref_1 */
  domain = isl_union_map_wrap(universe);
  if (read)
    external = isl_union_map_intersect_range(external, domain);
  else
    external = isl_union_map_intersect_domain(external, domain);
  external = isl_union_map_intersect_params(external,
              isl_set_copy(prog->scop->context));
  /* external contains flow dep that are associated with the group access */
  
  external = isl_union_map_subtract(external, overlap);
  /* external only contains access non-overlap RAW pairs */

//  // debug
//  p = isl_printer_print_union_map(p, external);
//  printf("\n");
//  // debug

  if (read) {
    tag_set = isl_union_map_range(external);
    external = wrapped_reference_to_access(tag_set, tagged); 
  } else {
    tag_set = isl_union_map_domain(external);
    external = wrapped_reference_to_access(tag_set, tagged);
  }

  if (empty < 0) 
    external = isl_union_map_free(external);
  else if (empty)
    external = isl_union_map_universe(external);

  access = isl_union_map_intersect(access, external); 
  empty = isl_union_map_is_empty(access);
  isl_union_map_free(access);

  if (empty)
    return isl_bool_true;
  else
    return isl_bool_false;
}

/* Return if the current module is valid to be generated. There are several cases to consider:
 * - For I/O group with all RAR depenendence, no copy-out modules to be generated.
 * - For I/O group with either RAW/RAR dependence, if the next read equals the previous write,
 *   no copy-in/copy-out to be generated.
 */
isl_bool is_module_valid(
  __isl_keep isl_schedule_node *node,  
  struct polysa_kernel *kernel, 
  struct polysa_array_ref_group *group, int read)
{
  int external_group = 1;

  if (group->group_type == POLYSA_PE_GROUP)
    return isl_bool_true;

  /* External group */
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    for (int j = 0; j < ref->n_io_info; j++) {
      struct polysa_io_info *io_info = ref->io_info[j];
      if (io_info->io_type == group->io_type && !isl_vec_cmp(io_info->dir, group->dir)) {
        if (io_info->dep->type != POLYSA_DEP_RAR) {
          external_group = 0;
          break;
        }
      }
    }
  }

  if (external_group) {
    if (read)
      return isl_bool_true;
    else
      return isl_bool_false;
  }  

  /* Internal group */
  if (internal_group_in_out_overlap(node, kernel, group, read))
    return isl_bool_false;

  return isl_bool_true;
}

static __isl_give isl_multi_pw_aff *extract_filter_size(
  struct polysa_kernel *kernel, __isl_take isl_union_set *domain, 
  __isl_keep isl_union_set *filter, isl_id_list *ids)
{
  int i, n;
  isl_set *grid;
  isl_set *context;
  isl_multi_pw_aff *size;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_set_get_ctx(filter), stdout);
//  p = isl_printer_print_union_set(p, filter);
//  printf("\n");
//  p = isl_printer_print_union_set(p, domain);
//  printf("\n");
//  // debug

  n = isl_id_list_n_id(ids);
  domain = isl_union_set_intersect(domain,
            isl_union_set_copy(filter));
//  // debug
//  p = isl_printer_print_union_set(p, domain);
//  printf("\n");
//  // debug
  grid = isl_union_set_params(domain);
  grid = isl_set_from_params(grid);
  grid = isl_set_add_dims(grid, isl_dim_set, n);
  for (i = 0; i < n; ++i) {
    int pos;
    isl_id *id;

    if (!grid)
      return NULL;

    id = isl_id_list_get_id(ids, i);
    pos = isl_set_find_dim_by_id(grid, isl_dim_param, id);
    isl_id_free(id);
    if (pos < 0)
      isl_die(isl_set_get_ctx(grid), isl_error_internal,
          "missing constraints on identifiers",
          grid = isl_set_free(grid));
    grid = isl_set_equate(grid, isl_dim_param, pos, isl_dim_set, i);
    grid = isl_set_project_out(grid, isl_dim_param, pos, 1);
  }

  grid = isl_set_coalesce(grid);
  size = ppcg_size_from_extent(grid);
  context = isl_set_params(isl_set_copy(kernel->context));
  return isl_multi_pw_aff_gist(size, context);
}

static char *generate_io_module_name(isl_ctx *ctx, struct polysa_array_ref_group *group, int level, int read) {
  isl_printer *p;

  p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, group->array->name);
  if (group->group_type == POLYSA_IO_GROUP) {
    if (group->local_array->n_io_group > 1) {
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
    }
  } else if (group->group_type == POLYSA_DRAIN_GROUP) {
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_str(p, "drain");
  }
  p = isl_printer_print_str(p, "_IO_L");
  p = isl_printer_print_int(p, level);
  if (read)
    p = isl_printer_print_str(p, "_in");
  else
    p = isl_printer_print_str(p, "_out");

  char *str = isl_printer_get_str(p);
  isl_printer_free(p);

  return str;
}

static __isl_give isl_schedule *generate_default_io_module_schedule(
  __isl_take struct polysa_hw_module *module, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int is_filter, int is_buffer, int read, int boundary) 
{
  isl_schedule *sched1, *sched2;
  isl_ctx *ctx;
  isl_printer *p;
  char *io_mark;
  int n_io_ids = 0;
  isl_id_list *io_ids;
  isl_id *id;
  int is_mark;
  isl_set *context;
  char *fifo_suffix, *buf_suffix;
  isl_union_set *empty_filter = NULL;
  isl_union_set *eq_filter = NULL;
  int depth;
  char *stmt_name;
  struct polysa_io_buffer *buf = NULL;
  isl_union_map *group_access;
  isl_union_set *group_domain;
  int i;

  ctx = isl_schedule_node_get_ctx(node);
  sched1 = isl_schedule_node_get_schedule(node);
  sched2 = isl_schedule_dup(sched1);
  isl_schedule_free(sched1);
  node = isl_schedule_get_root(sched2);
  isl_schedule_free(sched2);

  n_io_ids = space_dim - io_level + 1;
  io_ids = ppcg_scop_generate_names(gen->prog->scop, n_io_ids, "p"); 

//  // debug
//  isl_printer *pd = isl_printer_to_file(ctx, stdout);
//  pd = isl_printer_set_yaml_style(pd, ISL_YAML_STYLE_BLOCK);
//  // debug

  n_io_ids = 0;
  /* Update the context */
  context = isl_set_universe(isl_set_get_space(kernel->context));
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_union_map *umap;
      isl_union_set *uset;
      isl_multi_pw_aff *size;
      isl_id *id;
      isl_id_list *ids;

      umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
      uset = isl_union_map_range(umap);
      size = ppcg_size_from_extent(isl_set_from_union_set(uset));
      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));
      n_io_ids++;
      context = add_bounded_parameters_dynamic(context, size, ids);
      isl_id_list_free(ids);
      isl_multi_pw_aff_free(size);
    }
    node = isl_schedule_node_child(node, 0);
  }
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_context(node, context);

  /* Add the filters. */
  n_io_ids = 0;
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_id *id;
      isl_id_list *ids;
      isl_union_set *uset;

      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));      
      if (n_io_ids == space_dim - io_level) {
        if (is_filter) {
//          if (read)
//            uset = set_schedule_ge(node, ids);
//          else
//            uset = set_schedule_le(node, ids);         
          uset = set_schedule_ge(node, ids);
        } else {
          uset = set_schedule_eq(node, ids);
        }
      } else {
        uset = set_schedule_eq(node, ids);
      }
      n_io_ids++;
      node = isl_schedule_node_insert_filter(node, uset);
      isl_id_list_free(ids);
      node = isl_schedule_node_child(node, 0);
    }
    node = isl_schedule_node_child(node, 0);
  }
  if (module->to_pe) {
    /* Add filter to only send data to boundary PEs */
    while (!isl_schedule_node_is_io_mark(node, 1)) {
      if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
        isl_union_set *uset;

        if (read)
          uset = schedule_eq_lb(node);
        else
          uset = schedule_eq_ub(node);
        node = isl_schedule_node_insert_filter(node, uset);
        node = isl_schedule_node_child(node, 0);
      }
      node = isl_schedule_node_child(node, 0);
    }
  }
  node = polysa_tree_move_up_to_kernel(node);

  /* Add the data transfer statements */
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, io_level); 
  if (is_buffer && is_filter) {
    isl_id_list *ids;

    ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, space_dim - io_level));
    node = isl_schedule_node_parent(node);
    eq_filter = set_schedule_eq(node, ids); 
    node = isl_schedule_node_child(node, 0);
    
    isl_id_list_free(ids);
  }
  depth = isl_schedule_node_get_schedule_depth(node);
  /* Four types of I/O modules:
   * filter + no buffer
   * filter + buffer
   * no filter + no buffer
   * no filter + buffer
   */
  init_suffix(module, group, &fifo_suffix, &buf_suffix);
  /* Locate the next buffer */
  for (i = io_level; i >= 1; i--) {
    buf = group->io_buffers[i - 1];
    if (buf->tile != NULL)
      break;
  }
  if (is_buffer) {
    if (i != io_level) {
      /* The buffer is optimized out at this level. */
      is_buffer = 0;
    }
  }

  p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, read? "in_trans" : "out_trans");
  if (module->to_mem) 
    p = isl_printer_print_str(p, "_dram");
  if (boundary)
    p = isl_printer_print_str(p, "_boundary");
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_str(p, fifo_suffix);
  if (module->to_mem)
    p = isl_printer_print_str(p, "_local");
  p = isl_printer_print_str(p, is_filter == 0? ".0" : ".1");
  p = isl_printer_print_str(p, is_buffer == 0? ".0" : ".1");
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, depth - 1); 
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, space_dim - io_level);
  /* Insert the transfer statement */
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, buf->n_lane);

  /* Move the schedule node to the level of the buffer */
  node = polysa_tree_move_up_to_kernel(node);
  node = polysa_tree_move_down_to_depth(node, buf->tile->depth, kernel->core); 

  if (!buf->tile) {
    module->data_pack_inter = buf->n_lane;
    module->data_pack_intra = buf->n_lane;
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    /* Add the I/O statement for each array reference in the group. */
    node = add_io_copies_stmt_acc(kernel, group, node, buf->tile, buf->n_lane, read, stmt_name, read? 1: 0);
  } else {
    /* Add the I/O statement for the entire group. */
    module->data_pack_inter = buf->n_lane;
    module->data_pack_intra = buf->n_lane;
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    node = add_io_copies_stmt_tile(kernel, group, node, buf->tile, buf->tile, buf->n_lane, read, stmt_name, read? 1: 0, is_buffer);
    if (!is_buffer) {
      node = isl_schedule_node_cut(node);
      empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
      node = isl_schedule_node_insert_filter(node, empty_filter);
    }
  }

  if (is_buffer) {
    /* Add a filter node */     
    if (is_filter) {
      node = isl_schedule_node_insert_filter(node, eq_filter);  
      node = isl_schedule_node_child(node, 0);
    }

    /* Insert the extra transfer statement */
    p = isl_printer_to_str(ctx);
    p = isl_printer_print_str(p, read? "out_trans." : "in_trans.");
    p = isl_printer_print_str(p, fifo_suffix);
    p = isl_printer_print_str(p, "_local");
    p = isl_printer_print_str(p, ".0"); // filter
    p = isl_printer_print_str(p, ".1"); // buffer
    p = isl_printer_print_str(p, ".-1"); // sched_depth
    p = isl_printer_print_str(p, ".-1"); // param_id
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    /* Locate the next buffer after the current buffer */
    int cur_level = buf->level;
    struct polysa_io_buffer *cur_buf = buf;
    for (int i = cur_level - 1; i >= 1; i--) {
      buf = group->io_buffers[i - 1];
      if (buf->tile != NULL)
        break;
    }
//    if (cur_level > 1) {
//      buf = group->io_buffers[cur_level - 1 - 1];
//    }
    if (cur_level > 1) { 
      /* Move the schedule node to the level of the buffer */
      node = polysa_tree_move_down_to_io_mark(node, kernel->core, buf->level);
      node = isl_schedule_node_child(node, 0);
    }
    if (cur_level == 1 || !buf->tile) {
      p = isl_printer_print_str(p, ".");
      p = isl_printer_print_int(p, group->n_lane);
      stmt_name = isl_printer_get_str(p);
      isl_printer_free(p);
      module->data_pack_intra = group->n_lane; 
      node = add_io_copies_stmt_acc(kernel, group, node, cur_buf->tile, group->n_lane, read, stmt_name, read? 1 : 0); 
    } else {
      p = isl_printer_print_str(p, ".");
      p = isl_printer_print_int(p, buf->n_lane);
      stmt_name = isl_printer_get_str(p);
      isl_printer_free(p);
      module->data_pack_intra = buf->n_lane;
      node = add_io_copies_stmt_tile(kernel, group, node, cur_buf->tile, buf->tile, buf->n_lane, read, stmt_name, read? 1 : 0, is_buffer);
      node = isl_schedule_node_cut(node);
      empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
      node = isl_schedule_node_insert_filter(node, empty_filter);
    }
  }

//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  free(fifo_suffix);
  free(buf_suffix);

  /* Compute the union of domains of all the array references in the group. */
  group_access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    if (group->group_type == POLYSA_IO_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_io_group_ref_access_relation(group, ref, read, !read));
    } else if (group->group_type == POLYSA_DRAIN_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain));
    }
  }
  group_domain = isl_union_map_domain(group_access);
  group_domain = isl_union_set_coalesce(group_domain);
  /* Add the group domain as the filter */
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0); // context
  node = isl_schedule_node_child(node, 0); 
  node = isl_schedule_node_insert_filter(node, group_domain);

  /* Add the module mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  sched1 = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  if (!boundary) {
    module->sched = sched1;
    module->type = (group->group_type == POLYSA_DRAIN_GROUP)? DRAIN_MODULE : IO_MODULE;
    module->level = io_level;
    module->n_io_group++;
    module->io_groups = (struct polysa_array_ref_group **)realloc(module->io_groups,
        module->n_io_group * sizeof(struct polysa_array_ref_group *));
    module->io_groups[module->n_io_group - 1] = group;
    module->inst_ids = io_ids;
    module->kernel = kernel;
    module->is_buffer = is_buffer;
    module->is_filter = is_filter;
    if (read)
      module->in = 1;
    else
      module->in = 0;
    /* Create IO module variables */
    if (is_buffer) {
      for (int i = io_level; i >= 1; i--) {
        buf = group->io_buffers[i - 1];
        if (buf->tile != NULL)
          break;
      }
//      buf = group->io_buffers[io_level - 1];
      create_io_module_vars(module, kernel, buf->tile);
    }
  } else {
    isl_id_list_free(io_ids);
    module->boundary_sched = sched1;
  }

  return isl_stat_ok;
}

static __isl_give struct polysa_hw_module *generate_default_io_module(
  __isl_take struct polysa_hw_module *module, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int is_filter, int is_buffer, int read)
{
  isl_ctx *ctx = gen->ctx;
 
  // debug
//  isl_printer *pd = isl_printer_to_file(ctx, stdout);
//  pd = isl_printer_set_yaml_style(pd, ISL_YAML_STYLE_BLOCK);
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
  // debug

  generate_default_io_module_schedule(module, node, group, 
      kernel, gen, io_level, space_dim, is_filter, is_buffer, read, 0);

  if (is_filter) {
    /* Add the boundary module schedule */
    module->boundary = 1;
    generate_default_io_module_schedule(module, node, group,
        kernel, gen, io_level, space_dim, is_filter, is_buffer, read, 1);
  }

  return module;
}

static __isl_give isl_schedule *generate_io_module_outer(
  __isl_keep isl_schedule *sched, struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, 
  struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int read, int boundary)
{
  isl_ctx *ctx;
  int n_io_ids = 0;
  isl_id_list *io_ids;
  isl_id *id;
  isl_set *context;
  isl_union_set *empty_filter = NULL;
  char *stmt_name1, *stmt_name2, *stmt_name3, *stmt_name4, *stmt_name5;
  isl_union_map *group_access;
  isl_union_set *group_domain;
  isl_schedule_node *node, *graft1, *graft2, *graft3, *graft4, *graft5;
  isl_schedule *new_sched;
  int upper_io_level;
  isl_space *space;
  isl_union_set *domain;
  struct polysa_io_buffer *buf;

  new_sched = isl_schedule_dup(sched);
  node = isl_schedule_get_root(new_sched);
  isl_schedule_free(new_sched);
  ctx = isl_schedule_node_get_ctx(node);
 
//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug
  
  n_io_ids = space_dim - io_level + 1;
  io_ids = ppcg_scop_generate_names(gen->prog->scop, n_io_ids, "p"); 
  n_io_ids = 0;

  assert(module->to_mem == 0);
  upper_io_level = io_level + 1;

  /* Update the context */
  context = isl_set_universe(isl_set_get_space(kernel->context));
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_union_map *umap;
      isl_union_set *uset;
      isl_multi_pw_aff *size;
      isl_id *id;
      isl_id_list *ids;

      umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
      uset = isl_union_map_range(umap);
      size = ppcg_size_from_extent(isl_set_from_union_set(uset));
      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));
      n_io_ids++;
      context = add_bounded_parameters_dynamic(context, size, ids);
      isl_id_list_free(ids);
      isl_multi_pw_aff_free(size);
    }
    node = isl_schedule_node_child(node, 0);
  }
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_context(node, context);

  /* Add the filters. */
  n_io_ids = 0;
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, upper_io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_id *id;
      isl_id_list *ids;
      isl_union_set *uset;

      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));      
      uset = set_schedule_eq(node, ids);
      n_io_ids++;
      node = isl_schedule_node_insert_filter(node, uset);
      isl_id_list_free(ids);
      node = isl_schedule_node_child(node, 0);
    }
    node = isl_schedule_node_child(node, 0);
  }
  
  node = polysa_tree_move_up_to_kernel(node);

  /* Add the inter_trans and intra_trans function calls */
//  stmt_name1 = "io_module.inter_trans";
//  stmt_name2 = "io_module.intra_trans";
  stmt_name1 = boundary == 0? "io_module.inter_trans" : "io_module.inter_trans.boundary";
  stmt_name2 = "io_module.intra_trans";
  stmt_name3 = boundary == 0? "io_module.inter_intra" : "io_module.inter_intra.boundary";
  stmt_name4 = boundary == 0? "io_module.intra_inter" : "io_module.intra_inter.boundary"; 
  stmt_name5 = "io_module.state_handle";

  node = polysa_tree_move_down_to_io_mark(node, kernel->core, upper_io_level);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_cut(node);

  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name1);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft1 = isl_schedule_node_from_domain(domain);

  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name2);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft2 = isl_schedule_node_from_domain(domain);

  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name3);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft3 = isl_schedule_node_from_domain(domain);

  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name4);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft4 = isl_schedule_node_from_domain(domain);

  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name5);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft5 = isl_schedule_node_from_domain(domain);

//  if (read) {
//    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft1));
//    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft2));
//  } else {
//    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft2));
//    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft1));
//  } 
  if (read) {
    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft3));
  } else {
    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft4));
  }  
  if (gen->options->double_buffer) {
    /* Add misc statements for saving and switching states */
//    space = isl_space_set_alloc(ctx, 0, 0);
//    space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name3);
//    domain = isl_union_set_from_set(isl_set_universe(space));
//    graft3 = isl_schedule_node_from_domain(domain);
    node = isl_schedule_node_graft_before(node, isl_schedule_node_copy(graft5));
  }
  node = isl_schedule_node_cut(node);
  /* Insert an empty filter */
  empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  node = isl_schedule_node_insert_filter(node, empty_filter);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  if (gen->options->double_buffer) {
    /* Add the last function call */
    node = polysa_tree_move_up_to_kernel(node);
    node = isl_schedule_node_child(node, 0);
    node = isl_schedule_node_child(node, 0);
    if (read)
      node = isl_schedule_node_graft_after(node, isl_schedule_node_copy(graft2));
    else
      node = isl_schedule_node_graft_after(node, isl_schedule_node_copy(graft1));
  }
  isl_schedule_node_free(graft1);
  isl_schedule_node_free(graft2);
  isl_schedule_node_free(graft3);
  isl_schedule_node_free(graft4);
  isl_schedule_node_free(graft5);

  /* Compute the union of domains of all the array references in the group. */
  group_access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    if (group->group_type == POLYSA_IO_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_io_group_ref_access_relation(group, ref, read, !read));
    } else if (group->group_type == POLYSA_DRAIN_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain));
    }
  }
  group_domain = isl_union_map_domain(group_access);
  group_domain = isl_union_set_coalesce(group_domain);
  /* Add the group domain as the filter */
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0); // context
  node = isl_schedule_node_child(node, 0); 
  node = isl_schedule_node_insert_filter(node, group_domain);

  /* Add the module mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  new_sched = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  /* Update module information */
  if (!boundary) {
    module->type = (group->group_type == POLYSA_DRAIN_GROUP)? DRAIN_MODULE : IO_MODULE;
    module->level = io_level;
    module->n_io_group++;
    module->io_groups = (struct polysa_array_ref_group **)realloc(module->io_groups,
        module->n_io_group * sizeof(struct polysa_array_ref_group *));
    module->io_groups[module->n_io_group - 1] = group;
    module->inst_ids = io_ids;
    module->kernel = kernel;
    module->is_buffer = 1;
    module->is_filter = 1;
    if (read)
      module->in = 1;
    else
      module->in = 0;
    /* Create IO module variables */
    for (int i = io_level; i >= 1; i--) {
      buf = group->io_buffers[i - 1];
      if (buf->tile != NULL)
        break;
    }
    create_io_module_vars(module, kernel, buf->tile);
  } else {
    isl_id_list_free(io_ids);
  }

  return new_sched;
}

static __isl_give isl_schedule *generate_io_module_inter_trans(
  __isl_keep isl_schedule *sched, struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, 
  struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int read, int boundary)
{
  isl_schedule *new_sched;
  isl_ctx *ctx;
  isl_printer *p;
  char *io_mark;
  int n_io_ids = 0;
  isl_id_list *io_ids;
  isl_id *id;
  int is_mark;
  isl_set *context;
  char *fifo_suffix, *buf_suffix;
  isl_union_set *empty_filter = NULL;
  isl_union_set *eq_filter = NULL;
  int depth;
  char *stmt_name;
  struct polysa_io_buffer *buf = NULL;
  isl_union_map *group_access;
  isl_union_set *group_domain;
  isl_schedule_node *node;
  int upper_io_level;
  int is_filter = 1;
  int is_buffer = 1;
  int i;

  new_sched = isl_schedule_dup(sched);
  node = isl_schedule_get_root(new_sched);
  isl_schedule_free(new_sched);
  ctx = isl_schedule_node_get_ctx(node);
 
  n_io_ids = space_dim - io_level + 1;
  io_ids = ppcg_scop_generate_names(gen->prog->scop, n_io_ids, "p"); 
  n_io_ids = 0;

  assert(module->to_mem == 0);
  upper_io_level = io_level + 1;

  /* Update the context */
  context = isl_set_universe(isl_set_get_space(kernel->context));
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_union_map *umap;
      isl_union_set *uset;
      isl_multi_pw_aff *size;
      isl_id *id;
      isl_id_list *ids;

      umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
      uset = isl_union_map_range(umap);
      size = ppcg_size_from_extent(isl_set_from_union_set(uset));
      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));
      n_io_ids++;
      context = add_bounded_parameters_dynamic(context, size, ids);
      isl_id_list_free(ids);
      isl_multi_pw_aff_free(size);
    }
    node = isl_schedule_node_child(node, 0);
  }
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_context(node, context);

  /* Add the filters. */
  n_io_ids = 0;
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_id *id;
      isl_id_list *ids;
      isl_union_set *uset;

      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));      
      if (n_io_ids == space_dim - io_level) {
//        if (read)
//          uset = set_schedule_ge(node, ids);
//        else
//          uset = set_schedule_le(node, ids);         
        uset = set_schedule_ge(node, ids);
      } else {
        uset = set_schedule_eq(node, ids);
      }
      n_io_ids++;
      node = isl_schedule_node_insert_filter(node, uset);
      isl_id_list_free(ids);
      node = isl_schedule_node_child(node, 0);
    }
    node = isl_schedule_node_child(node, 0);
  }
  node = polysa_tree_move_up_to_kernel(node);

  /* Add the data transfer statements */
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, io_level); 
  depth = isl_schedule_node_get_schedule_depth(node);
  /* Four types of I/O modules:
   * filter + no buffer
   * filter + buffer
   * no filter + no buffer
   * no filter + buffer
   */
  init_suffix(module, group, &fifo_suffix, &buf_suffix);

  /* Locate the next buffer */
  for (i = io_level; i >= 1; i--) {
    buf = group->io_buffers[i - 1];
    if (buf->tile != NULL)
      break;
  }
  if (is_buffer) {
    if (i != io_level) {
      /* IO buffer is optimized out */
      is_buffer = 0;
    }
  }

  p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, read? "in_trans" : "out_trans");
  if (module->to_mem) 
    p = isl_printer_print_str(p, "_dram");
  if (boundary) 
    p = isl_printer_print_str(p, "_boundary");
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_str(p, fifo_suffix);
  if (module->to_mem)
    p = isl_printer_print_str(p, "_local");
  p = isl_printer_print_str(p, is_filter == 0? ".0" : ".1");
  p = isl_printer_print_str(p, is_buffer == 0? ".0" : ".1");
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, depth - 1); 
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, space_dim - io_level);
  /* Insert the transfer statement */
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, buf->n_lane);

  /* Move the schedule node to the level of the buffer */
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, buf->level); 
  node = isl_schedule_node_child(node, 0);
  if (!buf->tile) { 
    /* Add the I/O statement for each array reference in the group. */
    module->data_pack_inter = buf->n_lane;
    module->data_pack_intra = buf->n_lane;
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    node = add_io_copies_stmt_acc(kernel, group, node, buf->tile, buf->n_lane, read, stmt_name, read? 1: 0);
  } else {
    /* Add the I/O statement for the entire group. */
    module->data_pack_inter = buf->n_lane;
    module->data_pack_intra = buf->n_lane;
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    node = add_io_copies_stmt_tile(kernel, group, node, buf->tile, buf->tile, buf->n_lane, read, stmt_name, read? 1: 0, is_buffer);
    node = isl_schedule_node_cut(node);
    /* Insert empty filter */
    empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
    node = isl_schedule_node_insert_filter(node, empty_filter);
  }

//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  free(fifo_suffix);
  free(buf_suffix);

  /* Insert the function mark */
  node = polysa_tree_move_up_to_kernel(node);
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, upper_io_level);
  node = isl_schedule_node_child(node, 0);
  id = isl_id_alloc(ctx, "io_module.inter_trans", NULL);
  node = isl_schedule_node_insert_mark(node, id);

  /* Compute the union of domains of all the array references in the group. */
  group_access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    if (group->group_type == POLYSA_IO_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_io_group_ref_access_relation(group, ref, read, !read));
    } else if (group->group_type == POLYSA_DRAIN_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain));
    }
  }
  group_domain = isl_union_map_domain(group_access);
  group_domain = isl_union_set_coalesce(group_domain);
  /* Add the group domain as the filter */
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0); // context
  node = isl_schedule_node_child(node, 0); 
  node = isl_schedule_node_insert_filter(node, group_domain);

  /* Add the module mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  new_sched = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  isl_id_list_free(io_ids);

  return new_sched;
}

static __isl_give isl_schedule *generate_io_module_intra_trans(
  __isl_keep isl_schedule *sched, struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, 
  struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int read, int is_buffer)
{
  isl_ctx *ctx;
  isl_printer *p;
  char *io_mark;
  int n_io_ids = 0;
  isl_id_list *io_ids;
  isl_id_list *ids;
  isl_id *id;
  int is_mark;
  isl_set *context;
  char *fifo_suffix, *buf_suffix;
  isl_union_set *empty_filter = NULL;
  isl_union_set *eq_filter = NULL;
  int depth;
  char *stmt_name;
  struct polysa_io_buffer *buf = NULL;
  isl_union_map *group_access;
  isl_union_set *group_domain;
  isl_schedule *new_sched;
  isl_schedule_node *node;
  int upper_io_level;
  int i;

  new_sched = isl_schedule_dup(sched);
  node = isl_schedule_get_root(new_sched);
  isl_schedule_free(new_sched);
  ctx = isl_schedule_node_get_ctx(node);
 
  n_io_ids = space_dim - io_level + 1;
  io_ids = ppcg_scop_generate_names(gen->prog->scop, n_io_ids, "p"); 
  n_io_ids = 0;

  assert(module->to_mem == 0);
  upper_io_level = io_level + 1;

  /* Update the context */
  context = isl_set_universe(isl_set_get_space(kernel->context));
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_union_map *umap;
      isl_union_set *uset;
      isl_multi_pw_aff *size;
      isl_id *id;
      isl_id_list *ids;

      umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
      uset = isl_union_map_range(umap);
      size = ppcg_size_from_extent(isl_set_from_union_set(uset));
      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));
      n_io_ids++;
      context = add_bounded_parameters_dynamic(context, size, ids);
      isl_id_list_free(ids);
      isl_multi_pw_aff_free(size);
    }
    node = isl_schedule_node_child(node, 0);
  }
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_context(node, context);

  /* Add the filters. */
  n_io_ids = 0;
  node = polysa_tree_move_down_to_array(node, kernel->core);
  while (!isl_schedule_node_is_io_mark(node, upper_io_level)) {
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      isl_id *id;
      isl_id_list *ids;
      isl_union_set *uset;

      ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, n_io_ids));      
      uset = set_schedule_eq(node, ids);
      n_io_ids++;
      node = isl_schedule_node_insert_filter(node, uset);
      isl_id_list_free(ids);
      node = isl_schedule_node_child(node, 0);
    }
    node = isl_schedule_node_child(node, 0);
  }
  if (module->to_pe) {
    /* Add filter to only send data to boundary PEs */
    while (!isl_schedule_node_is_io_mark(node, 1)) {
      if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
        isl_union_set *uset;

        if (read)
          uset = schedule_eq_lb(node);
        else
          uset = schedule_eq_ub(node);
        node = isl_schedule_node_insert_filter(node, uset);
        node = isl_schedule_node_child(node, 0);
      }
      node = isl_schedule_node_child(node, 0);
    }
  }
  node = polysa_tree_move_up_to_kernel(node);

  /* Add the data transfer statements */
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, io_level); 

  ids = isl_id_list_from_id(isl_id_list_get_id(io_ids, space_dim - io_level));
  node = isl_schedule_node_parent(node);
  eq_filter = set_schedule_eq(node, ids); 
  node = isl_schedule_node_child(node, 0);
  
  isl_id_list_free(ids);
  init_suffix(module, group, &fifo_suffix, &buf_suffix);

  /* Add a filter node */     
  node = isl_schedule_node_parent(node);
  node = isl_schedule_node_insert_filter(node, eq_filter);  
  node = isl_schedule_node_child(node, 0);

  /* Locate the current buffer */
  for (i = io_level; i >= 1; i--) {
    buf = group->io_buffers[i - 1];
    if (buf->tile != NULL)
      break;
  }
  if (is_buffer) {
    if (i != io_level) {
      /* IO buffer is optimized out */
      is_buffer = 0;
    }
  }

  /* Insert the extra transfer statement */
  p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, read? "out_trans." : "in_trans.");
  p = isl_printer_print_str(p, fifo_suffix);
  p = isl_printer_print_str(p, "_local");
  p = isl_printer_print_str(p, ".0"); // filter
  p = isl_printer_print_str(p, is_buffer == 0? ".0" : ".1"); // buffer
  p = isl_printer_print_str(p, ".-1"); // sched_depth
  p = isl_printer_print_str(p, ".-1"); // param_id
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_int(p, buf->n_lane);

  /* Locate the next buffer after the current buffer */
  int cur_level = buf->level;
  struct polysa_io_buffer *cur_buf = buf;
  for (int i = cur_level - 1; i >= 1; i--) {
    buf = group->io_buffers[i - 1];
    if (buf->tile != NULL)
      break;
  }
//  if (cur_level > 1) {
//    buf = group->io_buffers[cur_level - 1 - 1];
//  }
  if (cur_level > 1) { 
    /* Move the schedule node to the level of the buffer */
    node = polysa_tree_move_down_to_io_mark(node, kernel->core, buf->level);
    node = isl_schedule_node_child(node, 0);
  }
  if (cur_level == 1 || !buf->tile) {
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, group->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    module->data_pack_intra = group->n_lane;
    node = add_io_copies_stmt_acc(kernel, group, node, cur_buf->tile, group->n_lane, read, stmt_name, read? 1: 0); 
  } else {
    p = isl_printer_print_str(p, ".");
    p = isl_printer_print_int(p, buf->n_lane);
    stmt_name = isl_printer_get_str(p);
    isl_printer_free(p);
    module->data_pack_intra = buf->n_lane;
    node = add_io_copies_stmt_tile(kernel, group, node, cur_buf->tile, buf->tile, buf->n_lane, read, stmt_name, read? 1: 0, is_buffer);
    node = isl_schedule_node_cut(node);
    /* Insert empty filter */
    empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
    node = isl_schedule_node_insert_filter(node, empty_filter);
  }
  
//  // debug
//  pd = isl_printer_print_schedule_node(pd, node);
//  pd = isl_printer_flush(pd);
//  // debug

  free(fifo_suffix);
  free(buf_suffix);

  /* Insert the function mark */
  node = polysa_tree_move_up_to_kernel(node);
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, upper_io_level);
  node = isl_schedule_node_child(node, 0);
  id = isl_id_alloc(ctx, "io_module.intra_trans", NULL);
  node = isl_schedule_node_insert_mark(node, id);

  /* Compute the union of domains of all the array references in the group. */
  group_access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; i++) {
    struct polysa_stmt_access *ref = group->refs[i];
    if (group->group_type == POLYSA_IO_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_io_group_ref_access_relation(group, ref, read, !read));
    } else if (group->group_type == POLYSA_DRAIN_GROUP) {
      group_access = isl_union_map_union(group_access,
          polysa_drain_group_ref_access_relation(group, ref, read, !read, kernel->expanded_domain));
    }
  }
  group_domain = isl_union_map_domain(group_access);
  group_domain = isl_union_set_coalesce(group_domain);
  /* Add the group domain as the filter */
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0); // context
  node = isl_schedule_node_child(node, 0); 
  node = isl_schedule_node_insert_filter(node, group_domain);

  /* Add the module mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  new_sched = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  isl_id_list_free(io_ids);

  return new_sched;
}

/* We will generate three seperate schedules for this type of I/O module.
 * Schedule 1: Outer loops contains two marks for inter_transfer and intra_transfer modules
 * Schedule 2: Inter_transfer function
 * Schedule 3: Intra_transfer function
 */
static __isl_give struct polysa_hw_module *generate_filter_buffer_io_module(
  __isl_take struct polysa_hw_module *module, __isl_keep isl_schedule_node *node,
  struct polysa_array_ref_group *group, struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int is_filter, int is_buffer, int read)
{
  isl_schedule *sched;
  isl_schedule *sched1, *sched2, *sched3;
  isl_schedule *boundary_sched2, *boundary_sched1;

  sched = isl_schedule_node_get_schedule(node);

  /* Inter transfer function */
  sched2 = generate_io_module_inter_trans(sched, module, group, kernel, gen,
      io_level, space_dim, read, 0);
  if (is_filter) {
    /* Add the boundary module schedule */
    module->boundary = 1;
    boundary_sched2 = generate_io_module_inter_trans(sched, module, group, kernel, gen,
        io_level, space_dim, read, 1);
  }
  /* Intra transfer function */
  sched3 = generate_io_module_intra_trans(sched, module, group, kernel, gen,
      io_level, space_dim, read, is_buffer);
  /* Outer loops */
  sched1 = generate_io_module_outer(sched, module, group, kernel, gen,
      io_level, space_dim, read, 0);
  if (is_filter) {
    /* Add the boundary module schedule */
    module->boundary = 1;
    boundary_sched1 = generate_io_module_outer(sched, module, group, kernel, gen,
        io_level, space_dim, read, 1);
  }

  isl_schedule_free(sched);

  module->sched = NULL;
  module->outer_sched = sched1;
  module->inter_sched = sched2;
  module->intra_sched = sched3;
  if (module->boundary) {
    module->boundary_outer_sched = boundary_sched1;
    module->boundary_inter_sched = boundary_sched2;
  }

  if (gen->options->double_buffer)
    module->double_buffer = 1;
  else
    module->double_buffer = 0;

  return module;
}

/* Generates the I/O modules for transffering the data.
 * The I/O module is decribed by two features:
 * - is_filter: If the module is a filter node, it will keep the data that belongs to it and 
 *   sends to the lower-level I/O modules or PEs. Else, it will simply pass the data to 
 *   downstream modules.
 * - is buffer: If the module is buffered. We will allocate a local buffer inside the module.
 */               
static __isl_give struct polysa_hw_module *generate_io_module_by_type(
  __isl_take struct polysa_hw_module *module, __isl_keep isl_schedule_node *node, 
  struct polysa_array_ref_group *group, struct polysa_kernel *kernel, struct polysa_gen *gen,
  int io_level, int space_dim, int is_filter, int is_buffer, int read)
{
  if (is_filter && is_buffer) {
    module = generate_filter_buffer_io_module(module, node, group, kernel, gen, io_level, space_dim, is_filter, is_buffer, read);
  } else {
    module = generate_default_io_module(module, node, group, kernel, gen, io_level, space_dim, is_filter, is_buffer, read);
  }

  return module;
}

/* For each I/O group, we will construct a set of I/O modules.
 * We inspect and transform the space loops. Starting from the innermost space loop,
 * we will cluster the I/O modules level by level.
 * Programmers can provide a configuration file specifying up to which level we should cluster the 
 * I/O modules, what the cluster direction we should use, and if we want to unroll this level of I/O modules.
 * Otherwise, we will use the pre-set policy, which is, we will cluster the I/O modules up to the
 * outermost level, and the I/O modules will be connected sequentially by default.
 * After the clustering, the outermost I/O modules will be unrolled to connected the external memory.
 * In the automatic mode, since all the modules are clustered, there is one single outermost I/O module. To support
 * multi-DRAM port/HBM, we also allow users to strip-mine the outermost clustered I/O loop again.
 */
__isl_give struct polysa_hw_module **sa_io_module_gen(
  struct polysa_array_ref_group *group,
  struct polysa_gen *gen, int *n_modules, int in, int out)
{
  // TODO: Add the support for manual tuning
  isl_schedule_node *node;
  isl_ctx *ctx;
  struct polysa_kernel *kernel;
  int space_dim;
  int io_level;
  struct polysa_hw_module **modules = NULL;
  int module_cnt = 0;
  int credit = 0;

  ctx = gen->ctx;
  node = isl_schedule_get_root(group->io_schedule);
  io_level = group->io_level;
  space_dim = group->space_dim;
  kernel = gen->kernel;

  /* Generate the I/O modules */
  node = polysa_tree_move_down_to_kernel(node);

  /* Test if the deps in this I/O group are carreied by array part loops.
   * If so, data hazards are possible, and we will set the credit as true
   * so that we could enable credit control between read and write I/O modules to 
   * prevent the data hazards. 
   */
  if (gen->options->credit_control) {
    if (group->local_array->array_type == POLYSA_INT_ARRAY) {
      isl_bool carried = isl_bool_false;
      isl_union_map *umap;
  
      node = polysa_tree_move_down_to_array(node, kernel->core);
      node = isl_schedule_node_parent(node);
      umap = isl_schedule_node_band_get_partial_schedule_union_map(node);
      for (int i = 0; i < group->n_ref; i++) {
        struct polysa_stmt_access *ref = group->refs[i];
        for (int j = 0; j < ref->n_io_info; j++) {
          struct polysa_io_info *io_info = ref->io_info[j];
          if (io_info->io_type == group->io_type && !isl_vec_cmp(io_info->dir, group->dir)) {
            isl_map *test;
            isl_map *schedule_dep;
            int dim;
            int is_parallel;
            isl_union_map *dep = isl_union_map_from_map(
                isl_map_factor_domain(
                isl_map_from_basic_map(isl_basic_map_copy(io_info->dep->isl_dep))));
  //          // debug
  //          isl_printer *p = isl_printer_to_file(gen->ctx, stdout);
  //          p = isl_printer_print_union_map(p, dep);
  //          printf("\n");
  //          p = isl_printer_print_union_map(p, umap);
  //          printf("\n");
  //          // debug
  
            dep = isl_union_map_apply_range(dep, isl_union_map_copy(umap));
            dep = isl_union_map_apply_domain(dep, isl_union_map_copy(umap));
  //          // debug
  //          p = isl_printer_print_union_map(p, dep);
  //          printf("\n");
  //          // debug
  
            if (isl_union_map_is_empty(dep)) {
              isl_union_map_free(dep);
              break;
            }
            schedule_dep = isl_map_from_union_map(dep);
            test = isl_map_universe(isl_map_get_space(schedule_dep));
            dim = isl_schedule_node_band_n_member(node);
            for (int n = 0; n < dim; n++) {
              test = isl_map_equate(test, isl_dim_in, n, isl_dim_out, n);
            }
            is_parallel = isl_map_is_subset(schedule_dep, test);
            isl_map_free(schedule_dep);
            isl_map_free(test);
  
            if (!is_parallel) { 
              carried = isl_bool_true;
              break;
            }
          }
        }
      }
      isl_union_map_free(umap); 
      if (carried) {
        credit = 1;
      }
      node = polysa_tree_move_up_to_kernel(node);
    }
  }

  /* For each I/O level, generate one I/O module */
  /* Copy-in group */
  if (in && is_module_valid(node, kernel, group, 1)) {
    group->array_io_dir = (group->array_io_dir == IO_OUT)? IO_INOUT : IO_IN;
    for (int i = io_level; i >= 1; i--) {
      struct polysa_hw_module *module;
      char *module_name = NULL;
      char *io_mark = NULL;
      isl_printer *p_str;
      int is_filter;
      int is_buffer;
      int innermost, outermost;
    
      /* Classify the module type */
      outermost = io_level;
      if (group->io_type == POLYSA_INT_IO)
        innermost = 1;
      else
        innermost = 2; // IO_L1 is integrated into PEs

      if (i == outermost)
        is_filter = 0;
      else
        is_filter = 1;

      if (group->group_type == POLYSA_DRAIN_GROUP) {
        if (i == innermost)
          is_buffer = 1;
        else
          is_buffer = 0;
      } else if (group->group_type == POLYSA_IO_GROUP) {
        if (group->local_array->array_type == POLYSA_INT_ARRAY) {
          if (group->io_type == POLYSA_EXT_IO) {
            if (i == innermost)
              is_buffer = 1;
            else
              is_buffer = 0;
          } else if (group->io_type == POLYSA_INT_IO) {
            is_buffer = 0;
          }
        } else if (group->local_array->array_type == POLYSA_EXT_ARRAY) {
          if (i == innermost)
            is_buffer = 1;
          else
            is_buffer = 0;
        }
      }

      if (gen->options->two_level_buffer) {
        /* When two-level buffering is enabled, we will implement a second-level buffer
         * at the outermost I/O module.
         */
        if (i == outermost)
          is_buffer = 1;
      }

      /* Generate the I/O module */
      if (i >= innermost && i <= outermost) {
        module = polysa_hw_module_alloc();
        module_name = generate_io_module_name(ctx, group, i, 1);
        module->name = module_name;
        module->to_pe = (i == innermost)? 1 : 0;
        module->to_mem = (i == outermost)? 1 : 0;
        module->credit = (i == outermost)? credit : 0;
        module->n_array_ref = group->local_array->n_io_group_refs;
        if (module->to_mem)
          group->local_array->n_io_group_refs++;

        module = generate_io_module_by_type(module, node, group, kernel, 
            gen, i, space_dim, is_filter, is_buffer, 1);

        module_cnt++;
        modules = (struct polysa_hw_module **)realloc(modules,
            module_cnt * sizeof(struct polysa_hw_module *));
        modules[module_cnt - 1] = module;
      }
    }
  }

  /* Copy-out group */
  if (out && is_module_valid(node, kernel, group, 0)) {
    group->array_io_dir = (group->array_io_dir == IO_IN)? IO_INOUT : IO_OUT;
    for (int i = 1; i <= io_level; i++) {
      struct polysa_hw_module *module;
      char *module_name = NULL;
      char *io_mark = NULL;
      isl_printer *p_str;
      int is_filter;
      int is_buffer;
      int innermost, outermost;
    
      /* Classify the module type */
      outermost = io_level;
      if (group->io_type == POLYSA_INT_IO)
        innermost = 1;
      else
        innermost = 2; // IO_L1 is integrated into PEs

      if (i == outermost)
        is_filter = 0;
      else
        is_filter = 1;
      if (group->group_type == POLYSA_DRAIN_GROUP) {
        if (i == innermost)
          is_buffer = 1;
        else
          is_buffer = 0;
      } else if (group->group_type == POLYSA_IO_GROUP) {
        if (group->io_type == POLYSA_INT_IO) 
          is_buffer = 0;
        else {
          if (i == innermost) 
            is_buffer = 1;
          else
            is_buffer = 0;
        }
      }

      if (gen->options->two_level_buffer) {
        /* When two-level buffering is enabled, we will implement a second-level buffer
         * at the outermost I/O module.
         */
        if (i == outermost)
          is_buffer = 1;
      }

      /* Generate the I/O module */
      if (i >= innermost && i <= outermost) {
        module = polysa_hw_module_alloc();
        module_name = generate_io_module_name(ctx, group, i, 0);
        module->name = module_name;
        module->to_pe = (i == innermost)? 1 : 0;
        module->to_mem = (i == outermost)? 1 : 0;
        module->credit = (i == outermost)? credit : 0;
        module->n_array_ref = group->local_array->n_io_group_refs;
        if (module->to_mem)
          group->local_array->n_io_group_refs++;

        module = generate_io_module_by_type(module, node, group, kernel, 
            gen, i, space_dim, is_filter, is_buffer, 0);

        module_cnt++;
        modules = (struct polysa_hw_module **)realloc(modules,
            module_cnt * sizeof(struct polysa_hw_module *));
        modules[module_cnt - 1] = module;
      } 
    }
  }

  isl_schedule_node_free(node);
  *n_modules = module_cnt;
  return modules;
}

static __isl_give isl_schedule *pe_dummy_gen_module_call(struct polysa_gen *gen,
  struct polysa_pe_dummy_module *pe_dummy_module)
{
  struct polysa_array_ref_group *group;
  isl_schedule *sched;
  isl_schedule_node *node;
  struct polysa_kernel *kernel;
  struct polysa_hw_module *module;
  int n_member;
  isl_union_set *L1_filter;
  isl_bool insert_L1 = isl_bool_false;
  isl_printer *p_str;
  isl_ctx *ctx;
  char *stmt_name;
  isl_id *id;
  isl_union_map *prefix, *extension;
  isl_union_set *domain, *range;

  module = pe_dummy_module->module;
  kernel = module->kernel;
  ctx = gen->ctx;
  group = pe_dummy_module->io_group;
  sched = isl_schedule_dup(group->io_L1_schedule);
  node = isl_schedule_get_root(sched);
  isl_schedule_free(sched);
  isl_space *space;
  isl_union_set *empty_filter;
  isl_schedule_node *graft;

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  /* Delete the node above the array mark */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_parent(node);
//  while (isl_schedule_node_get_type(node) != isl_schedule_node_mark) {
  while (!polysa_tree_node_is_kernel(node)) {
    node = isl_schedule_node_delete(node);
    node = isl_schedule_node_parent(node);
  }

  node = polysa_tree_move_down_to_mark(node, kernel->core, "io_L1");
  node = isl_schedule_node_parent(node);
  n_member = isl_schedule_node_band_n_member(node);
  if (n_member > 1) {
    node = isl_schedule_node_band_split(node, n_member - 1);
    node = isl_schedule_node_child(node, 0);
  }
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    L1_filter = schedule_eq_ub(node);
    insert_L1 = isl_bool_true;
  }
  
//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  node = polysa_tree_move_down_to_mark(node, kernel->core, "io_L1");
  node = isl_schedule_node_child(node, 0);
  if (insert_L1) {
    node = isl_schedule_node_insert_filter(node, L1_filter);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  /* Delete the node under the pe mark */
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  node = isl_schedule_node_cut(node);

  /* Graft an extension node */
  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
              isl_union_pw_multi_aff_copy(kernel->contraction));
  domain = isl_union_map_range(prefix);

  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, "module_call.");
//  p_str = isl_printer_print_str(p_str, module->name);
  p_str = polysa_array_ref_group_print_prefix(group, p_str);
  p_str = isl_printer_print_str(p_str, "_PE_dummy");
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name);
  free(stmt_name);

  isl_point *pnt = isl_point_zero(space);
  isl_set *set = isl_set_from_point(pnt);
  range = isl_union_set_from_set(isl_set_copy(set));
  extension = isl_union_map_from_domain_and_range(domain, range);
  graft = isl_schedule_node_from_extension(extension);

  isl_map *map = isl_set_identity(set);
  map = isl_map_reset_tuple_id(map, isl_dim_out);
  isl_union_map *umap = isl_union_map_from_map(map);
  isl_multi_union_pw_aff *mupa = isl_multi_union_pw_aff_from_union_map(umap);

  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  node = isl_schedule_node_graft_before(node, graft);

  /* Insert an empty filter */
  empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  node = isl_schedule_node_insert_filter(node, empty_filter);

  /* Add module mark after the kernel mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  /* Add pe_dummy module mark after the module mark */
  id = isl_id_alloc(ctx, "pe_dummy_module", pe_dummy_module);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  sched = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  return sched;
}

static isl_stat top_module_pe_gen_module_call(struct polysa_gen *gen,
  struct polysa_hw_top_module *top, struct polysa_hw_module *module)
{
  isl_schedule *schedule;
  isl_schedule_node *node, *graft;
  isl_id *id;
  struct polysa_kernel *kernel = gen->kernel;
  isl_space *space;
  isl_ctx *ctx;
  isl_union_set *domain;
  isl_union_set *empty_filter;
  isl_printer *p_str;
  char *stmt_name;

  schedule = gen->schedule;
  schedule = isl_schedule_dup(schedule);
  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);
  ctx = isl_schedule_node_get_ctx(node);

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  /* Delete the node above the array mark */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_parent(node);
//  while (isl_schedule_node_get_type(node) != isl_schedule_node_mark) {
  while (!polysa_tree_node_is_kernel(node)) {
    node = isl_schedule_node_delete(node);
    node = isl_schedule_node_parent(node);
  }

  /* Delete the node under the pe mark */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  node = split_band(node, kernel->n_sa_dim);

  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_cut(node);

  /* Graft an extension node */
  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, "module_call.");
  p_str = isl_printer_print_str(p_str, module->name);
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name);
  free(stmt_name);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft = isl_schedule_node_from_domain(domain);

  node = isl_schedule_node_graft_before(node, graft);

  /* Insert an empty filter */
  empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  node = isl_schedule_node_insert_filter(node, empty_filter);

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Add module mark after the kernel mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  top->n_module_calls++;
  top->module_call_scheds = (isl_schedule **)realloc(top->module_call_scheds,
      top->n_module_calls * sizeof(isl_schedule *));
  top->module_call_scheds[top->n_module_calls - 1] = schedule;
//  top->module_call_names = (char **)realloc(top->module_call_names,
//      top->n_module_calls * sizeof(char *));
//  top->module_call_names[top->n_module_calls - 1] = strdup(module->name);

  if (module->n_pe_dummy_modules > 0) {
    /* Generate dummy module calls */
    for (int i = 0; i < module->n_pe_dummy_modules; i++) {
      struct polysa_pe_dummy_module *pe_dummy_module;
      isl_schedule *sched;

      pe_dummy_module = module->pe_dummy_modules[i];
      sched = pe_dummy_gen_module_call(gen, pe_dummy_module);

      top->n_module_calls++;
      top->module_call_scheds = (isl_schedule **)realloc(top->module_call_scheds,
          top->n_module_calls * sizeof(isl_schedule *));
      top->module_call_scheds[top->n_module_calls - 1] = sched;
      
//      isl_printer *p_str = isl_printer_to_str(ctx);
//      p_str = polysa_array_ref_group_print_prefix(pe_dummy_module->io_group, p_str);
//      p_str = isl_printer_print_str(p_str, "_PE_dummy");
//      char *module_name = isl_printer_get_str(p_str);
//      isl_printer_free(p_str);
//      top->module_call_names = (char **)realloc(top->module_call_names,
//        top->n_module_calls * sizeof(char *));
//      top->module_call_names[top->n_module_calls - 1] = module_name;
    }
  }

  return isl_stat_ok;
}

static isl_stat top_module_pe_gen_fifo_decl(struct polysa_gen *gen, 
  struct polysa_hw_top_module *top, struct polysa_hw_module *module)
{
  isl_schedule *schedule;
  isl_schedule_node *node, *graft;
  isl_id *id;
  struct polysa_kernel *kernel = gen->kernel;
  isl_space *space;
  isl_ctx *ctx = gen->ctx;
  isl_union_set *domain;
  isl_union_set *empty_filter;
  isl_printer *p_str;
  char *stmt_name;
 
//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  for (int i = 0; i < module->n_io_group; i++) {
    struct polysa_array_ref_group *group = module->io_groups[i];
    isl_multi_aff *io_trans;
    isl_mat *io_trans_mat;
    isl_id *id;
    isl_union_set *L1_filter = NULL;
    bool insert_L1 = isl_bool_false;

    schedule = isl_schedule_dup(group->io_L1_schedule); 
    node = isl_schedule_get_root(schedule);
    isl_schedule_free(schedule);

//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug

    /* Delete the node above the array mark */
    node = polysa_tree_move_down_to_array(node, kernel->core);
    node = isl_schedule_node_parent(node);
//    while (isl_schedule_node_get_type(node) != isl_schedule_node_mark) {
    while (!polysa_tree_node_is_kernel(node)) {
      node = isl_schedule_node_delete(node);
      node = isl_schedule_node_parent(node);
    }

    if (group->pe_io_dir == IO_INOUT) {
      int n_member;
      node = polysa_tree_move_down_to_mark(node, kernel->core, "io_L1");
      node = isl_schedule_node_parent(node);
      n_member = isl_schedule_node_band_n_member(node);
      node = isl_schedule_node_band_split(node, n_member - 1);
      node = isl_schedule_node_child(node, 0);
//      // debug
//      p = isl_printer_print_schedule_node(p, node);
//      printf("\n");
//      // debug
      if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
        L1_filter = schedule_eq_ub(node);
        insert_L1 = isl_bool_true;
      }
      node = polysa_tree_move_up_to_array(node);
    }
    
    /* Delete the node under the pe mark */
    node = polysa_tree_move_down_to_pe(node, kernel->core);
    node = isl_schedule_node_cut(node);

    /* Graft an extension node */
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, "fifo_decl.");
    p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
    stmt_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    space = isl_space_set_alloc(ctx, 0, 0);
    id = isl_id_alloc(ctx, stmt_name, group);
    space = isl_space_set_tuple_id(space, isl_dim_set, id);
    free(stmt_name);
    domain = isl_union_set_from_set(isl_set_universe(space));
    graft = isl_schedule_node_from_domain(domain);

    node = isl_schedule_node_graft_before(node, graft);

    if (insert_L1) {
      isl_set *set;
      isl_multi_union_pw_aff *mupa;
      isl_union_map *prefix;
      isl_union_set *domain;
      isl_union_set *range;
      isl_union_map *extension;
      isl_map *map;
      isl_union_map *umap;

      /* Graft an extension node for boundary PE */
      node = isl_schedule_node_insert_filter(node, L1_filter);
      node = isl_schedule_node_child(node, 0);
//      // debug
//      p = isl_printer_print_schedule_node(p, node);
//      printf("\n");
//      // debug

      prefix = isl_schedule_node_get_prefix_schedule_relation(node);
      prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
                  isl_union_pw_multi_aff_copy(kernel->contraction));
      domain = isl_union_map_range(prefix); 

      p_str = isl_printer_to_str(ctx);
      p_str = isl_printer_print_str(p_str, "fifo_decl_boundary.");
      p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
      stmt_name = isl_printer_get_str(p_str);
      isl_printer_free(p_str);
      space = isl_space_set_alloc(ctx, 0, 1);
      id = isl_id_alloc(ctx, stmt_name, group);
      space = isl_space_set_tuple_id(space, isl_dim_set, id);
      free(stmt_name);
  
      isl_point *pnt = isl_point_zero(space);
      set = isl_set_from_point(pnt); 
      range = isl_union_set_from_set(isl_set_copy(set));

      extension = isl_union_map_from_domain_and_range(domain, range);
 //     // debug
 //     p = isl_printer_print_union_map(p, extension);
 //     printf("\n");
 //     // debug
      graft = isl_schedule_node_from_extension(extension);

      map = isl_set_identity(set);
      map = isl_map_reset_tuple_id(map, isl_dim_out);
      umap = isl_union_map_from_map(map);
      mupa = isl_multi_union_pw_aff_from_union_map(umap);
 //     // debug
 //     p = isl_printer_print_multi_union_pw_aff(p, mupa);
 //     printf("\n");
 //     // debug

      graft = isl_schedule_node_child(graft, 0);
      graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

      while (graft && isl_schedule_node_has_parent(graft)) 
        graft = isl_schedule_node_parent(graft);

      node = isl_schedule_node_graft_before(node, graft);     
    } else {
      isl_union_set_free(L1_filter);
    }

    /* Insert an empty filter */
    empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
    node = isl_schedule_node_insert_filter(node, empty_filter);

//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug

    /* Add module mark after the kernel mark */
    id = isl_id_alloc(ctx, "module", module);
    node = polysa_tree_move_up_to_kernel(node);
    node = isl_schedule_node_child(node, 0);
    node = isl_schedule_node_insert_mark(node, id);

    schedule = isl_schedule_node_get_schedule(node);
    isl_schedule_node_free(node);

    top->n_fifo_decls++;
    top->fifo_decl_scheds = (isl_schedule **)realloc(top->fifo_decl_scheds,
        top->n_fifo_decls * sizeof(isl_schedule *));
    top->fifo_decl_scheds[top->n_fifo_decls - 1] = schedule;
    top->fifo_decl_names = (char **)realloc(top->fifo_decl_names,
        top->n_fifo_decls * sizeof(char *));
    /* Generate fifo_decl name in the format of 
     * [fifo_name].[fifo_width] 
     */
    p_str = isl_printer_to_str(ctx);
    p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
    p_str = isl_printer_print_str(p_str, "_");
    p_str = isl_printer_print_str(p_str, module->name);
    p_str = isl_printer_print_str(p_str, ".");
    int n_lane = get_io_group_n_lane(module, group);
    int data_size = group->array->size;
    int width = data_size * n_lane; // in bytes
    p_str = isl_printer_print_int(p_str, width);
    top->fifo_decl_names[top->n_fifo_decls - 1] = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
  }

  return isl_stat_ok;
}

/* Delete the node under the space loop */
static isl_stat top_module_pe_gen(struct polysa_gen *gen, struct polysa_hw_top_module *top,
  struct polysa_hw_module *module)
{
  /* Generate the function call schedule */
  top_module_pe_gen_module_call(gen, top, module);

  /* Generate the fifo declaration schedule */
  top_module_pe_gen_fifo_decl(gen, top, module);

  return isl_stat_ok;
}

/* The input "node" points to the node below io_[module->level] mark.
 * Return the node points to the kernel mark.
 */
static __isl_give isl_schedule_node *io_gen_module_call(
  __isl_take isl_schedule_node *node, struct polysa_hw_module *module,
  struct polysa_kernel *kernel, struct polysa_array_ref_group *group,
  int boundary)
{
  isl_printer *p_str;
  char *stmt_name;
  isl_space *space;
  isl_union_set *domain, *empty_filter, *lower_level_filter;
  isl_schedule_node *graft;
  isl_bool insert_lower = isl_bool_false;
  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  isl_id *id;
  isl_union_map *prefix, *extension, *umap;
  isl_union_set *range;
  isl_set *set;
  isl_map *map;
  isl_multi_union_pw_aff *mupa;

  /* Collect the filter for the lower I/O module */
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    if (module->level > 1) {
      lower_level_filter = schedule_eq_lb(node);
      insert_lower = isl_bool_true;
    }
  }
  
  /* Graft an extension node for module call. */
  prefix = isl_schedule_node_get_prefix_schedule_relation(node);
  prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
      isl_union_pw_multi_aff_copy(kernel->contraction));
  domain = isl_union_map_range(prefix);

  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, "module_call_upper.");
  p_str = isl_printer_print_str(p_str, module->name);
  if (boundary)
    p_str = isl_printer_print_str(p_str, ".boundary");
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  space = isl_space_set_alloc(ctx, 0, 0);
  space = isl_space_set_tuple_name(space, isl_dim_set, stmt_name);
  free(stmt_name);

  isl_point *pnt = isl_point_zero(space);
  set = isl_set_from_point(pnt);  
  range = isl_union_set_from_set(isl_set_copy(set));

  extension = isl_union_map_from_domain_and_range(domain, range);
  graft = isl_schedule_node_from_extension(extension);

  map = isl_set_identity(set);
  map = isl_map_reset_tuple_id(map, isl_dim_out);
  umap = isl_union_map_from_map(map);
  mupa = isl_multi_union_pw_aff_from_union_map(umap);

  graft = isl_schedule_node_child(graft, 0);
  graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

  while (graft && isl_schedule_node_has_parent(graft))
    graft = isl_schedule_node_parent(graft);

  node = isl_schedule_node_graft_before(node, graft);

  if (module->level > 1) {
    node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level - 1);
  }
  node = isl_schedule_node_cut(node);

  /* Graft an extension node for lower level transfer */
  if (insert_lower) {
    node = isl_schedule_node_insert_filter(node, lower_level_filter);
    node = isl_schedule_node_child(node, 0);
  }
  {
    isl_union_map *prefix;
    isl_union_set *domain, *range;
    isl_point *pnt;
    isl_set *set;
    isl_union_map *extension, *umap;
    isl_map *map;
    isl_multi_union_pw_aff *mupa;

    prefix = isl_schedule_node_get_prefix_schedule_relation(node);
    prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
                isl_union_pw_multi_aff_copy(kernel->contraction));
    domain = isl_union_map_range(prefix);  
    
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, "module_call_lower.");
    p_str = isl_printer_print_str(p_str, module->name);
    if (boundary)
      p_str = isl_printer_print_str(p_str, ".boundary");

    stmt_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    space = isl_space_set_alloc(ctx, 0, 0);
    id = isl_id_alloc(ctx, stmt_name, group);
    space = isl_space_set_tuple_id(space, isl_dim_set, id);
    free(stmt_name);

    pnt = isl_point_zero(space);
    set = isl_set_from_point(pnt);
    range = isl_union_set_from_set(isl_set_copy(set));

    extension = isl_union_map_from_domain_and_range(domain, range);
    graft = isl_schedule_node_from_extension(extension);

    map = isl_set_identity(set);
    map = isl_map_reset_tuple_id(map, isl_dim_out);
    umap = isl_union_map_from_map(map);
    mupa = isl_multi_union_pw_aff_from_union_map(umap);

    graft = isl_schedule_node_child(graft, 0);
    graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

    while (graft && isl_schedule_node_has_parent(graft))
      graft = isl_schedule_node_parent(graft);

    node = isl_schedule_node_graft_after(node, graft);
  }

  /* Insert an empty filter */
  empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  node = isl_schedule_node_insert_filter(node, empty_filter);

  node = polysa_tree_move_up_to_kernel(node);

  return node;
}

static isl_stat top_module_io_gen_module_call(
  struct polysa_gen *gen, struct polysa_hw_top_module *top,
  struct polysa_hw_module *module,
  struct polysa_array_ref_group *group) 
{
  isl_schedule *schedule;
  isl_ctx *ctx = gen->ctx;
  isl_schedule_node *node, *graft;
  isl_id *id;
  struct polysa_kernel *kernel = gen->kernel;
  isl_printer *p_str;
  char *stmt_name;
  isl_space *space;
  isl_union_set *domain, *empty_filter, *lower_level_filter;
  isl_bool insert_lower = isl_bool_false;
  int boundary = module->boundary;
  isl_union_set *boundary_filter, *non_boundary_filter;
  isl_union_set_list *boundary_filters;

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  // debug

  /* Transform the schedule */
  schedule = isl_schedule_dup(group->io_schedule);
  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);

  /* Delete the node above the array mark */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_parent(node);
  while (!polysa_tree_node_is_kernel(node)) {
    node = isl_schedule_node_delete(node);
    node = isl_schedule_node_parent(node);
  }

  /* Collect the filter for the boundary and non-boundary I/O module */
  if (boundary) {
    node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level);
    node = isl_schedule_node_parent(node);
    if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
      boundary_filter = schedule_eq_ub(node);
      non_boundary_filter = schedule_neq_ub(node);
      boundary_filters = isl_union_set_list_from_union_set(non_boundary_filter);
      boundary_filters = isl_union_set_list_add(boundary_filters, boundary_filter);

      node = isl_schedule_node_child(node, 0); // io_mark
      node = isl_schedule_node_child(node, 0); // band
      node = isl_schedule_node_insert_sequence(node, boundary_filters);
      /* The node now is right below the io_[module->level] mark */
    }
  } else {
    node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level);
    node = isl_schedule_node_child(node, 0);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  p = isl_printer_flush(p);
//  // debug

  if (boundary) {
    node = isl_schedule_node_child(node, 0); // filter
    node = isl_schedule_node_child(node, 0); // band
    /* non-boundary */
//    // debug
//    p = isl_printer_print_schedule_node(p, node);
//    printf("\n");
//    // debug
    node = io_gen_module_call(node, module, kernel, group, 0);

    node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level);
    node = isl_schedule_node_child(node, 0); // sequence
    node = isl_schedule_node_child(node, 1); // filter
    node = isl_schedule_node_child(node, 0); // band
    /* boundary */
    node = io_gen_module_call(node, module, kernel, group, 1);
  } else {
    node = io_gen_module_call(node, module, kernel, group, 0);
  }

//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Add module mark after the kernel mark */
  id = isl_id_alloc(ctx, "module", module);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  top->n_module_calls++;
  top->module_call_scheds = (isl_schedule **)realloc(top->module_call_scheds,
      top->n_module_calls * sizeof(isl_schedule *));
  top->module_call_scheds[top->n_module_calls - 1] = schedule;

  return isl_stat_ok;
}

/* Currently only works for filter I/O */
static isl_stat top_module_io_gen_fifo_decl(struct polysa_gen *gen,
  struct polysa_hw_top_module *top,
  struct polysa_hw_module *module, struct polysa_array_ref_group *group)
{
  isl_schedule *schedule;
  isl_schedule_node *node, *graft;
  isl_union_set *filter = NULL, *empty_filter;
  struct polysa_kernel *kernel = gen->kernel;
  bool insert_filter = isl_bool_false;
  char *stmt_name;
  isl_space *space;
  isl_union_set *domain;
  isl_printer *p_str;
  isl_id *id;
  isl_ctx *ctx = gen->ctx;

  if (module->to_mem) 
    return isl_stat_ok;

  schedule = isl_schedule_dup(group->io_schedule);
  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);

  /* Delete the node above the array mark */
  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_parent(node);
//  while (isl_schedule_node_get_type(node) != isl_schedule_node_mark) {
  while (!polysa_tree_node_is_kernel(node)) {
    node = isl_schedule_node_delete(node);
    node = isl_schedule_node_parent(node);
  }
 
  node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level);
  node = isl_schedule_node_parent(node);  
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    filter = schedule_eq_ub(node);
    insert_filter = isl_bool_true;
  }
  node = polysa_tree_move_up_to_array(node);

  node = polysa_tree_move_down_to_io_mark(node, kernel->core, module->level);
  node = isl_schedule_node_cut(node);

  /* Graft an extension node */
  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, "fifo_decl.");
  p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
  stmt_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  space = isl_space_set_alloc(ctx, 0, 0);
  id = isl_id_alloc(ctx, stmt_name, group);
  space = isl_space_set_tuple_id(space, isl_dim_set, id);
  free(stmt_name);
  domain = isl_union_set_from_set(isl_set_universe(space));
  graft = isl_schedule_node_from_domain(domain);

  node = isl_schedule_node_graft_before(node, graft);

  if (insert_filter) {
    isl_union_map *prefix, *extension, *umap;
    isl_union_set *domain, *range;
    isl_point *pnt;
    isl_set *set;
    isl_map *map;
    isl_multi_union_pw_aff *mupa;

    node = isl_schedule_node_insert_filter(node, filter);
    node = isl_schedule_node_child(node, 0);
    
    prefix = isl_schedule_node_get_prefix_schedule_relation(node);
    prefix = isl_union_map_preimage_domain_union_pw_multi_aff(prefix,
                isl_union_pw_multi_aff_copy(kernel->contraction));
    domain = isl_union_map_range(prefix);

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, "fifo_decl_boundary.");
    p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
    stmt_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    space = isl_space_set_alloc(ctx, 0, 1);
    id = isl_id_alloc(ctx, stmt_name, group);
    space = isl_space_set_tuple_id(space, isl_dim_set, id);
    free(stmt_name);

    pnt = isl_point_zero(space);
    set = isl_set_from_point(pnt);
    range = isl_union_set_from_set(isl_set_copy(set));

    extension = isl_union_map_from_domain_and_range(domain, range);
    graft = isl_schedule_node_from_extension(extension);
    map = isl_set_identity(set);
    map = isl_map_reset_tuple_id(map, isl_dim_out);
    umap = isl_union_map_from_map(map);
    mupa = isl_multi_union_pw_aff_from_union_map(umap);

    graft = isl_schedule_node_child(graft, 0);
    graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

    while (graft && isl_schedule_node_has_parent(graft))
      graft = isl_schedule_node_parent(graft);

    node = isl_schedule_node_graft_before(node, graft);
  }

  /* Insert an empty filter */
  empty_filter = isl_union_set_from_set(isl_set_empty(isl_set_get_space(kernel->context)));
  node = isl_schedule_node_insert_filter(node, empty_filter);

  /* Add module mark after the kernel mark */
  id = isl_id_alloc(ctx, "module", module);
  node = polysa_tree_move_up_to_kernel(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_mark(node, id);

  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);

  top->n_fifo_decls++;
  top->fifo_decl_scheds = (isl_schedule **)realloc(top->fifo_decl_scheds,
      top->n_fifo_decls * sizeof(isl_schedule *));
  top->fifo_decl_scheds[top->n_fifo_decls - 1] = schedule;
  top->fifo_decl_names = (char **)realloc(top->fifo_decl_names,
      top->n_fifo_decls * sizeof(char *));
  /* Generate fifo_decl name in the format of
   * [fifo_name].[fifo_width]
   */
  p_str = isl_printer_to_str(ctx);
  p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
  p_str = isl_printer_print_str(p_str, "_");
  p_str = isl_printer_print_str(p_str, module->name);
  p_str = isl_printer_print_str(p_str, ".");
  int n_lane = get_io_group_n_lane(module, group);
  int data_size = group->array->size;
  int width = data_size * n_lane; // in bytes
  p_str = isl_printer_print_int(p_str, width);
  top->fifo_decl_names[top->n_fifo_decls - 1] = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  return isl_stat_ok; 

}

static isl_stat top_module_io_gen(struct polysa_gen *gen, struct polysa_hw_top_module *top,
  struct polysa_hw_module *module)
{
  struct polysa_array_ref_group *group;
  assert(module->n_io_group == 1);
  group = module->io_groups[0];

  /* Generate the function call schedule */
  top_module_io_gen_module_call(gen, top, module, group);

  /* Generate the fifo declaration schedule */
  top_module_io_gen_fifo_decl(gen, top, module, group);

  return isl_stat_ok;
}

/* Generate the scheduel for module calls and fifo declarations */
__isl_give struct polysa_hw_top_module *sa_top_module_gen(struct polysa_gen *gen)
{
  struct polysa_hw_top_module *top_module;

  top_module = polysa_hw_top_module_alloc();
  top_module->hw_modules = gen->hw_modules;
  top_module->kernel = gen->kernel;
  top_module->n_hw_modules = gen->n_hw_modules;

  for (int i = 0; i < gen->n_hw_modules; i++) {
    struct polysa_hw_module *module = gen->hw_modules[i];
    if (module->type == PE_MODULE) {      
      top_module_pe_gen(gen, top_module, gen->hw_modules[i]);
    } else {
      top_module_io_gen(gen, top_module, gen->hw_modules[i]);
    }
  }

  return top_module;
}

/* The input modules are organized in the sequence of:
 * PE module
 * I/O module (copy-in and copy-out)
 * Drain module
 * We will reorder the modules following the below sequence:
 * I/O module (copy-in)
 * PE module
 * I/O module (copy-out)
 * Drain module
 * The reason for the re-ordering is for CSim in Xilinx environment.
 */
static __isl_give struct polysa_hw_module **hw_module_reorder(
  __isl_take struct polysa_hw_module **modules, int n_module)
{
  struct polysa_hw_module **modules_new = (struct polysa_hw_module **)malloc(n_module *
      sizeof(struct polysa_hw_module *));
  int pos = 0;

  /* I/O module (copy-in) */
  for (int i = 0; i < n_module; i++) {
    struct polysa_hw_module *module = modules[i];
    if (module->type == IO_MODULE && module->in) {
      modules_new[pos] = module;
      pos++;
    }
  }

  /* PE module */
  modules_new[pos] = modules[0];
  pos++;

  /* I/O module (copy-out) */
  for (int i = 0; i < n_module; i++) {
    struct polysa_hw_module *module = modules[i];
    if (module->type == IO_MODULE && !module->in) {
      modules_new[pos] = module;
      pos++;
    }
  }

  /* Drain module */
  for (int i = 0; i < n_module; i++) {
    struct polysa_hw_module *module = modules[i];
    if (module->type == DRAIN_MODULE) {
      modules_new[pos] = module;
      pos++;
    }
  }

  free(modules);
  return modules_new;
}

static cJSON *load_tuning_config(char *config_file)
{
  FILE *f;
  char *buffer = NULL;
  cJSON *config = NULL;
  long length;

  f = fopen(config_file, "rb");
  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = malloc(length + 1);
    if (buffer) {
      buffer[length] = '\0';
      fread(buffer, 1, length, f);
    }
    fclose(f);
  } else {
    printf("[PolySA] Error: open file: %s\n", config_file);
  }

  if (buffer) {
    config = cJSON_Parse(buffer);
    free(buffer);
  }

  return config;
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
  isl_schedule *hw_schedule;
  struct polysa_kernel *kernel;
  isl_id *id;
  cJSON *tuning_config = NULL;

  /* Load the tuning configuration file */
  tuning_config = load_tuning_config(gen->options->config);
  if (!tuning_config) {
    isl_schedule_free(schedule);
    printf("[PolySA] Error: PolySA configuration file not found: %s\n", gen->options->config);
    exit(1);
  }
  gen->tuning_config = tuning_config;
    
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

  /* Perform stages include:
   * Array Generation
   * PE Optimization
   * Data Transfer Optimization 
   */
  node = mark_kernels(gen, node);
  
  id = isl_schedule_node_mark_get_id(node);
  kernel = (struct polysa_kernel *)isl_id_get_user(id);
  isl_id_free(id);
  schedule = isl_schedule_node_get_schedule(node);

  /* Build new schedules for each hardware components.
   * The total number of schedules = 
   * 1. the default schedule (cpu code)
   * 2. PE schedule
   * 3. I/O module schedule
   * 4. drain schedule
   * 5. top module schedule
   */
  gen->schedule = schedule;
  gen->n_hw_modules = 1;
  gen->hw_modules = isl_calloc_array(gen->ctx, struct polysa_hw_module *, gen->n_hw_modules);
  gen->hw_modules[0] = NULL;
  /* IO module */
  for (int i = 0; i < kernel->n_array; i++) {
    struct polysa_local_array_info *info = &kernel->array[i];
    info->n_io_group_refs = 0;
    for (int j = 0; j < info->n_io_group; j++) {
      int n_hw_modules = 0;
      struct polysa_hw_module **hw_modules;
      hw_modules = sa_io_module_gen(info->io_groups[j], gen, &n_hw_modules, 1, 1);
      
      gen->hw_modules = (struct polysa_hw_module **)realloc(gen->hw_modules, 
          (gen->n_hw_modules + n_hw_modules) * sizeof(struct polysa_hw_module *));
      for (int k = 0; k < n_hw_modules; k++) {
        gen->hw_modules[gen->n_hw_modules + k] = hw_modules[k];
      }
      gen->n_hw_modules += n_hw_modules;
      if (hw_modules)
        free(hw_modules);
    }
  }
  for (int i = 0; i < kernel->n_array; i++) {
    struct polysa_local_array_info *info = &kernel->array[i];
    if (!info->drain_group)
      continue;
    int n_hw_modules = 0;
    struct polysa_hw_module **hw_modules;
    hw_modules = sa_io_module_gen(info->drain_group, gen, &n_hw_modules, 0, 1);

    if (n_hw_modules > 0) {
      gen->hw_modules = (struct polysa_hw_module **)realloc(gen->hw_modules, 
          (gen->n_hw_modules + n_hw_modules) * sizeof(struct polysa_hw_module *));
      for (int j = 0; j < n_hw_modules; j++) {
        gen->hw_modules[gen->n_hw_modules + j] = hw_modules[j];
      }
      gen->n_hw_modules += n_hw_modules;
    }
    if (hw_modules)
      free(hw_modules);
  }
  /* PE */
  gen->hw_modules[0] = sa_pe_module_gen(gen); 

  /* Reorder the sequence of the modules */
  gen->hw_modules = hw_module_reorder(gen->hw_modules, gen->n_hw_modules); 

  /* top module */
  struct polysa_hw_top_module *top_module = sa_top_module_gen(gen); 
  gen->hw_top_module = top_module;

  node = sa_add_copies(gen, node); 
  node = sa_add_to_from_device(node, domain, prefix, gen->prog); 
  node = isl_schedule_node_root(node);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_insert_guard(node, guard);
  node = isl_schedule_node_child(node, 0);

  node = sa_add_init_clear_device(node); 

  isl_schedule_free(gen->schedule);
  gen->schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  cJSON_Delete(gen->tuning_config);

  return gen->schedule;
}

isl_stat sa_top_module_generate_code(struct polysa_gen *gen) 
{
  struct polysa_hw_top_module *top = gen->hw_top_module;
  /* fifo declaration */
  top->fifo_decl_trees = (isl_ast_node **)malloc(
    top->n_fifo_decls * sizeof(isl_ast_node *));
  for (int i = 0; i < top->n_fifo_decls; i++) {
    top->fifo_decl_trees[i] = sa_fifo_decl_generate_code(gen,
        top->fifo_decl_scheds[i]);
  }

  /* module call */
  top->module_call_trees = (isl_ast_node **)malloc(
    top->n_module_calls * sizeof(isl_ast_node *));
  for (int i = 0; i < top->n_module_calls; i++) {
    top->module_call_trees[i] = sa_module_call_generate_code(gen,
        top->module_call_scheds[i]);
  }

  return isl_stat_ok;
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
    gen->schedule = sa_map_to_device(gen, schedule);

    /* Generate the AST tree. */    
    gen->tree = sa_generate_code(gen, gen->schedule);
    for (int i = 0; i < gen->n_hw_modules; i++) {
      if (gen->hw_modules[i]->is_filter == 1 && 
          gen->hw_modules[i]->is_buffer == 1) {
        sa_filter_buffer_io_module_generate_code(gen, gen->hw_modules[i]);
      } else {
        sa_module_generate_code(gen, gen->hw_modules[i]); 
      }
    }
    sa_top_module_generate_code(gen);

    /* Extract loop structure for latency estimation */
    for (int i = 0; i < gen->n_hw_modules; i++) {
      sa_extract_loop_info(gen, gen->hw_modules[i]);
    }
    /* Dump out the array information */
    sa_extract_array_info(gen->kernel);
    /* Extract design information for resource estimation */
    sa_extract_design_info(gen);

    p = ppcg_set_macro_names(p);
    p = ppcg_print_exposed_declarations(p, prog->scop);
    p = gen->print(p, gen->prog, gen->tree, gen->hw_modules, gen->n_hw_modules, gen->hw_top_module,
          &gen->types, gen->print_user);
    
    isl_ast_node_free(gen->tree);
    polysa_kernel_free(gen->kernel);

    for (int i = 0; i < gen->n_hw_modules; i++) {
      polysa_hw_module_free(gen->hw_modules[i]);
    }
    free(gen->hw_modules);
    polysa_hw_top_module_free(gen->hw_top_module);
//    // TODO: valgrind
//    isl_schedule_free(gen->schedule);
  }

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
    struct polysa_prog *prog, __isl_keep isl_ast_node *trees, 
    struct polysa_hw_module **modules, int n_module,
    struct polysa_hw_top_module *module,
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
  gen.hw_modules = NULL;
  gen.n_hw_modules = 0;
  gen.hw_top_module = NULL;
  gen.schedule = NULL;
  gen.kernel = NULL;
  gen.tuning_config = NULL;

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
