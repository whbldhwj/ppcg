#include "polysa_group.h"
#include "polysa_codegen.h"
#include "polysa_array_tile.h"
#include "polysa_tree.h"

/* Internal data structure for polysa_group_references.
 */
struct polysa_group_data {
  struct polysa_gen *gen;
	struct ppcg_scop *scop;
  /* The schedule depth where the kernel launch will be 
   * introduced.
   */
	int kernel_depth;
  /* The schedule depth at which the copying in/from local_memory
   * is computed. The copy operation may then later
   * be hoisted to a higher level.
   */
	int local_depth;
  int pe_depth;
	// int thread_depth;
	// int n_thread;
	// isl_set *privatization;
  isl_schedule *schedule;

  /* All the schedules are formulated in terms of the original statement
   * instances, i.e., those that appear in the domains of the access 
   * relations. 
   */
  /* Contains the kernel_depth dimensions of the host schedule. */
	isl_union_map *host_sched;
  /* Contains the first local_depth dimensions of the kernel schedule. */
	isl_union_map *local_sched;
  /* Contains the first local_depth dimensions of the kernel schedule. */
	isl_union_map *copy_sched;
	// isl_union_map *thread_sched;
  isl_union_map *pe_sched;
  /* A union map representation of the entire kernel schedule. */
	isl_union_map *full_sched;
};

/* Return the prefix schedule at "node" as a relation
 * between domain elements and schedule dimensions after detecting
 * equalities in this relation.
 */
static __isl_give isl_union_map *prefix_with_equalities(
  __isl_keep isl_schedule_node *node)
{
  isl_union_map *schedule;

  schedule = isl_schedule_node_get_prefix_schedule_relation(node);
  /* Simplify */
  schedule = isl_union_map_detect_equalities(schedule);

  return schedule;
}

/* Expand the domain of the schedule "s" by plugging in
 * the contraction "contraction" and return the result.
 */
static isl_union_map *expand(__isl_take isl_union_map *s,
  __isl_keep isl_union_pw_multi_aff *contraction)
{
  contraction = isl_union_pw_multi_aff_copy(contraction);
  s = isl_union_map_preimage_domain_union_pw_multi_aff(s, contraction);
  return s;
}

/* Fill up the groups array with singleton groups, i.e., one group
 * per reference, initializing the array, access, write, n_ref and refs fields.
 * In particular the access field is initialized to the scheduled
 * access relation of the array reference.
 *
 * Return the number of elements initialized, i.e., the number of
 * active references in the current kernel.
 */
static int populate_array_references_pe(struct polysa_local_array_info *local,
	struct polysa_array_ref_group **groups, struct polysa_group_data *data)
{
	int i;
  int j;
	int n;
	isl_ctx *ctx = isl_union_map_get_ctx(data->pe_sched);

	n = 0;
	for (i = 0; i < local->array->n_ref; ++i) {
    isl_union_map *umap;
		isl_map *map;
		struct polysa_array_ref_group *group;
		struct polysa_stmt_access *access = local->array->refs[i];

		map = isl_map_copy(access->access);
		umap = isl_union_map_from_map(map);
		umap = isl_union_map_apply_domain(umap,
			isl_union_map_copy(data->pe_sched));

  	if (isl_union_map_is_empty(umap)) {
		  isl_union_map_free(umap);
		  continue;
		}

		map = isl_map_from_union_map(umap);
		map = isl_map_detect_equalities(map);

		group = isl_calloc_type(ctx, struct polysa_array_ref_group);
		if (!group) {
		  isl_map_free(map);
		  return -1;
		}
		group->local_array = local;
		group->array = local->array;
		group->access = map;
		group->write = access->write;
		group->exact_write = access->exact_write;
		group->slice = access->n_index < local->array->n_index;
		group->refs = &local->array->refs[i];
		group->n_ref = 1;
    group->io_type = POLYSA_UNKNOWN_IO;
    group->dir = NULL;
    group->group_type = POLYSA_PE_GROUP;
    group->local_tile = NULL;
    group->io_trans = NULL;
//    group->io_trans_mat = NULL;
    group->io_pe_expr = NULL;
    group->n_io_buffer = 0;
    group->io_buffers = NULL;
    group->copy_schedule = NULL;

		groups[n++] = group;
	}

	return n;
}

/* Fill up the groups array with singleton groups, i.e., one group
 * per reference, initializing the array, access, write, n_ref and refs fields.
 * In particular the access field is initialized to the scheduled
 * access relation of the array reference.
 *
 * Return the number of elements initialized, i.e., the number of
 * active references in the current kernel.
 */
static int populate_array_references_io(struct polysa_local_array_info *local,
	struct polysa_array_ref_group **groups, struct polysa_group_data *data)
{
	int i;
  int j;
	int n;
	isl_ctx *ctx = isl_union_map_get_ctx(data->pe_sched);

	n = 0;
	for (i = 0; i < local->array->n_ref; ++i) {
    for (j = 0; j < local->array->refs[i]->n_io_info; ++j) {    
      isl_union_map *umap;
		  isl_map *map;
		  struct polysa_array_ref_group *group;
		  struct polysa_stmt_access *access = local->array->refs[i];

		  map = isl_map_copy(access->access);
		  umap = isl_union_map_from_map(map);
		  umap = isl_union_map_apply_domain(umap,
				isl_union_map_copy(data->copy_sched));

  		if (isl_union_map_is_empty(umap)) {
			  isl_union_map_free(umap);
			  continue;
		  }

		  map = isl_map_from_union_map(umap);
		  map = isl_map_detect_equalities(map);

		  group = isl_calloc_type(ctx, struct polysa_array_ref_group);
		  if (!group) {
			  isl_map_free(map);
			  return -1;
		  }
		  group->local_array = local;
		  group->array = local->array;
		  group->access = map; // not used
		  group->write = access->write;
		  group->exact_write = access->exact_write;
		  group->slice = access->n_index < local->array->n_index;
		  group->refs = &local->array->refs[i];
		  group->n_ref = 1;
      group->io_type = access->io_info[j]->io_type;
      group->dir = isl_vec_copy(access->io_info[j]->dir);
      group->group_type = POLYSA_IO_GROUP;
      group->pe_io_dir = IO_NULL;
      group->array_io_dir = IO_NULL;
      group->io_trans = NULL;
//      group->io_trans_mat = NULL;
      group->io_pe_expr = NULL;
      group->io_L1_pe_expr = NULL;
      group->n_io_buffer = 0;
      group->io_buffers = NULL;
      group->copy_schedule = NULL;

		  groups[n++] = group;
    }
	}

	return n;
}

/* Combine the given two groups into a single group, containing
 * the references of both groups.
 */
static struct polysa_array_ref_group *join_groups(
	struct polysa_array_ref_group *group1,
	struct polysa_array_ref_group *group2)
{
	int i;
	isl_ctx *ctx;
	struct polysa_array_ref_group *group;

	if (!group1 || !group2)
		return NULL;

	ctx = isl_map_get_ctx(group1->access);
	group = isl_calloc_type(ctx, struct polysa_array_ref_group);
	if (!group)
		return NULL;
	group->local_array = group1->local_array;
	group->array = group1->array;
	group->access = isl_map_union(isl_map_copy(group1->access),
					isl_map_copy(group2->access));
	group->write = group1->write || group2->write;
	group->exact_write = group1->exact_write && group2->exact_write;
	group->slice = group1->slice || group2->slice;
	group->n_ref = group1->n_ref + group2->n_ref;
	group->refs = isl_alloc_array(ctx, struct polysa_stmt_access *,
					group->n_ref);
	if (!group->refs)
		return polysa_array_ref_group_free(group);
	for (i = 0; i < group1->n_ref; ++i)
		group->refs[i] = group1->refs[i];
	for (i = 0; i < group2->n_ref; ++i)
		group->refs[group1->n_ref + i] = group2->refs[i];

  group->io_type = group1->io_type;
  group->dir = isl_vec_copy(group1->dir);
  group->group_type = group1->group_type;
  group->pe_io_dir = group1->pe_io_dir;
  group->array_io_dir = group1->array_io_dir;
  group->io_trans = group1->io_trans;
//  group->io_trans_mat = group1->io_trans_mat;
  group->io_pe_expr = group1->io_pe_expr;
  group->io_L1_pe_expr = group1->io_L1_pe_expr;
  group->n_io_buffer = group1->n_io_buffer;
  group->io_buffers = group1->io_buffers;

	return group;
}

/* Combine the given two groups into a single group and free
 * the original two groups.
 */
static struct polysa_array_ref_group *join_groups_and_free(
	struct polysa_array_ref_group *group1,
	struct polysa_array_ref_group *group2)
{
	struct polysa_array_ref_group *group;

	group = join_groups(group1, group2);
	polysa_array_ref_group_free(group1); 
	polysa_array_ref_group_free(group2);
	return group;
}

/* Return the union of all read (read = 1) and/or write (write = 1)
 * access relations in the group.
 */
__isl_give isl_union_map *polysa_array_ref_group_access_relation(
	struct polysa_array_ref_group *group, int read, int write)
{
	int i;
	isl_union_map *access;

	access = isl_union_map_empty(isl_map_get_space(group->access));
	for (i = 0; i < group->n_ref; ++i) {
		isl_map *map_i;

		if (!((read && group->refs[i]->read) ||
		     (write && group->refs[i]->write)))
			continue;
		map_i = isl_map_copy(group->refs[i]->access);
		access = isl_union_map_union(access,
					    isl_union_map_from_map(map_i));
	}

	return access;
}

/* Replace the host schedule dimensions in the access relation "access"
 * by parameters, so that they are treated as fixed when checking for reuse
 * (within a kernel) or whether two consecutive elements are accessed
 * (within a kernel).
 */
static __isl_give isl_union_map *localize_access(struct polysa_group_data *data,
	__isl_take isl_union_map *access)
{
	int n;
	isl_space *space;
	isl_set *param;
	isl_union_map *umap;
	isl_id_list *ids;

	umap = isl_union_map_copy(data->host_sched);
	space = isl_union_map_get_space(umap);
	n = data->kernel_depth;
	ids = ppcg_scop_generate_names(data->scop, n, "__ppcg_host_");
  /* Add "n" isl_dim_in from "0" that equates the parameters "ids". */
	param = parametrization(space, n, 0, ids);
	isl_id_list_free(ids);
	umap = isl_union_map_intersect_range(umap,
						isl_union_set_from_set(param));
	access = isl_union_map_intersect_domain(access,
						isl_union_map_domain(umap));

	return access;
}

///* Check if the given access is coalesced (or if there is no point
// * in trying to coalesce the access by mapping the array to local memory).
// * That is, check whether incrementing the dimension that will get
// * wrapped over the last thread index results in incrementing
// * the last array index.
// *
// * If no two consecutive array elements are ever accessed by "access",
// * then mapping the corresponding array to shared memory will not
// * improve coalescing.  In fact, the copying will likely be performed
// * by a single thread.  Consider the access as coalesced such that
// * the caller will not try and map the array to shared memory just
// * to improve coalescing.
// *
// * This function is only called for access relations without reuse and
// * kernels with at least one thread identifier.
// */
//static int access_is_coalesced(struct polysa_group_data *data,
//	__isl_keep isl_union_map *access)
//{
//	int dim;
//	isl_space *space;
//	isl_set *accessed;
//	isl_map *access_map;
//	isl_map *next_thread_x;
//	isl_map *next_element;
//	isl_map *map;
//	int coalesced, empty;
//
//	access = isl_union_map_copy(access);
//	access = isl_union_map_apply_domain(access,
//				isl_union_map_copy(data->full_sched));
//	access_map = isl_map_from_union_map(access);
//
//	space = isl_map_get_space(access_map);
//	space = isl_space_range(space);
//	dim = isl_space_dim(space, isl_dim_set);
//	if (dim == 0)
//		next_element = isl_map_empty(isl_space_map_from_set(space));
//	else
//		next_element = next(space, dim - 1);
//
//	accessed = isl_map_range(isl_map_copy(access_map));
//	map = isl_map_copy(next_element);
//	map = isl_map_intersect_domain(map, isl_set_copy(accessed));
//	map = isl_map_intersect_range(map, accessed);
//	empty = isl_map_is_empty(map);
//	isl_map_free(map);
//
//	if (empty < 0 || empty) {
//		isl_map_free(next_element);
//		isl_map_free(access_map);
//		return empty;
//	}
//
//	space = isl_map_get_space(access_map);
//	space = isl_space_domain(space);
//	next_thread_x = next(space, data->thread_depth + data->n_thread - 1);
//
//	map = isl_map_apply_domain(next_thread_x, isl_map_copy(access_map));
//	map = isl_map_apply_range(map, access_map);
//
//	coalesced = isl_map_is_subset(map, next_element);
//
//	isl_map_free(next_element);
//	isl_map_free(map);
//
//	return coalesced;
//}

/* Given the schedule node "node" that points to "kernel" mark, and the data transfer vector "dir",
 * modify the space band of the schedule to reflect the data transfer scheme. 
 * Return the new schedule with depth down to the PE ray level.
 *
 * When modifying the schedule, we are performing a space-time transformation on the space band.
 * Specifically, by choosing "dir" as the projection vector d , and the same "dir" as the scheduling vector s.
 * We first build the transformation matrix, then apply it to the partial schedule of the schedule node.
 *
 * The transformation matrix is composed of:
 * | P |
 *  ---
 * | S |
 *
 * Pd^T = 0
 * Given a projection vector d with length of n, P is composed of (n - 1) basises from the null space of d.
 * Since rank(null(d)) + rank(d) = n, and rank(d) = 1, therefore, rank(null(d)) = n - 1
 * We will first compute the null space of d and then use them to compose the matrix P
 * S = s.
 *
 * The pe_ray_sched_depth = array_depth + rank(P)
 */
__isl_give isl_schedule *get_io_schedule(__isl_take isl_schedule *schedule, __isl_keep isl_vec *dir, __isl_give isl_multi_aff **io_trans, __isl_give isl_mat **io_trans_mat, int hbm)
{
  isl_mat *trans_mat, *dir_mat, *null_mat;
  isl_ctx *ctx = isl_schedule_get_ctx(schedule);
  int space_dim;
  isl_schedule_node *node;
  struct polysa_kernel *kernel;
  isl_id *id;
  isl_multi_union_pw_aff *space_sched;
  isl_multi_aff *ma;
  int can_use_hbm;

  /* Sink to the space band. */
  node = isl_schedule_get_root(schedule);
  node = polysa_tree_move_down_to_kernel(node);
  id = isl_schedule_node_mark_get_id(node);
  kernel = isl_id_get_user(id);
  isl_id_free(id);

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  space_dim = isl_schedule_node_band_n_member(node);

  /* Build the transformation matrix */ 
  trans_mat = isl_mat_alloc(ctx, space_dim, space_dim);
  dir_mat = isl_mat_alloc(ctx, 1, space_dim);
  for (int i = 0; i < isl_vec_size(dir); i++) {
    dir_mat = isl_mat_set_element_val(dir_mat, 0, i, 
        isl_vec_get_element_val(dir, i));
  }
  null_mat = isl_mat_right_kernel(dir_mat);

  for (int i = 0; i < isl_mat_cols(null_mat); i++)
    for (int j = 0; j < isl_mat_rows(null_mat); j++) {
      trans_mat = isl_mat_set_element_val(trans_mat, i, j, 
          isl_mat_get_element_val(null_mat, j, i));
    }
  for (int i = 0; i < isl_vec_size(dir); i++) {
    trans_mat = isl_mat_set_element_val(trans_mat, isl_mat_cols(null_mat), i,
        isl_vec_get_element_val(dir, i));
  }
  *io_trans_mat = trans_mat;

  /* Modify the partial schedule of the space band. */
  space_sched = isl_schedule_node_band_get_partial_schedule(node);
  /* Convert the transformation matrix to a isl_multi_aff type */
  isl_space *domain_space = isl_multi_union_pw_aff_get_space(space_sched);
  isl_space *space = isl_space_map_from_set(domain_space);
  ma = isl_multi_aff_identity(space);

  for (int i = 0; i < isl_mat_rows(trans_mat); i++) {
    isl_aff *aff = isl_multi_aff_get_aff(ma, i);
    for (int j = 0; j < isl_mat_cols(trans_mat); j++) {
      isl_val *val = isl_mat_get_element_val(trans_mat, i, j);
      aff = isl_aff_set_coefficient_si(aff, isl_dim_in, j, isl_val_get_num_si(val));
      isl_val_free(val);
    }
    ma = isl_multi_aff_set_aff(ma, i, aff);
  }

  space_sched = isl_multi_union_pw_aff_apply_multi_aff(space_sched, isl_multi_aff_copy(ma));
  *io_trans = ma;

  node = isl_schedule_node_delete(node);
  node = isl_schedule_node_insert_partial_schedule(node, space_sched);

  if (isl_mat_cols(null_mat) > 0)
    can_use_hbm = 1;
  else
    can_use_hbm = 0;

  node = isl_schedule_node_band_split(node, isl_mat_cols(null_mat)); // inter_ray
  
  id = isl_id_alloc(ctx, "io_L2", NULL);
  node = isl_schedule_node_insert_mark(node, id);
  node = isl_schedule_node_child(node, 0);

  if (isl_mat_cols(null_mat) > 0)
    node = isl_schedule_node_child(node, 0); // intra_ray
  id = isl_id_alloc(ctx, "io_L1", NULL);
  node = isl_schedule_node_insert_mark(node, id);


  isl_schedule_free(schedule);
  schedule = isl_schedule_node_get_schedule(node);
  node = isl_schedule_node_free(node);
  isl_mat_free(null_mat);
//  isl_mat_free(trans_mat);

  return schedule;
}

/* Return the prefix I/O schedule at io_level "level". */
static __isl_give isl_union_map *get_io_schedule_at_level(__isl_keep isl_schedule *sched, int level)
{
  isl_schedule_node *node;
  struct polysa_kernel *kernel;
  isl_id *id;
  isl_union_map *io_sched;

  node = isl_schedule_get_root(sched);
  
  node = polysa_tree_move_down_to_kernel(node);
  id = isl_schedule_node_mark_get_id(node);
  kernel = isl_id_get_user(id);
  isl_id_free(id);

  node = polysa_tree_move_down_to_io_mark(node, kernel->core, level);

  io_sched = prefix_with_equalities(node);
  io_sched = expand(io_sched, kernel->contraction);

  isl_schedule_node_free(node);

  return io_sched;
}

//static __isl_give isl_union_map *get_io_pe_schedule(__isl_take isl_schedule *sched, __isl_keep isl_vec *dir)
//{
//  isl_schedule_node *node;
//  struct polysa_kernel *kernel;
//  isl_id *id;
//  isl_union_map *io_sched;
//  isl_multi_aff *io_trans;
//  isl_mat *io_trans_mat;
//
//  sched = get_io_schedule(sched, dir, &io_trans, &io_trans_mat, 0);
//  node = isl_schedule_get_root(sched);
//  isl_schedule_free(sched);
//
//  node = polysa_tree_move_down_to_kernel(node);
//  id = isl_schedule_node_mark_get_id(node);
//  kernel = isl_id_get_user(id);
//  isl_id_free(id);
//
//  node = polysa_tree_move_down_to_mark(node, kernel->core, "pe");
//
//  io_sched = prefix_with_equalities(node);
//  io_sched = expand(io_sched, kernel->contraction);
//
//  isl_schedule_node_free(node);
//  isl_multi_aff_free(io_trans);
//  isl_mat_free(io_trans_mat);
//
//  return io_sched;
//}

/* Map the domain of "access" to the outer data->pe_depth
 * schedule dimensions.   
 */
static __isl_give isl_map *local_access_pe(struct polysa_array_ref_group *group,
	__isl_keep isl_union_map *access, struct polysa_group_data *data)
{
	isl_union_map *local;

  local = isl_union_map_copy(access);
  /* Group at the PE level. */
  local = isl_union_map_apply_domain(local,
      isl_union_map_copy(data->pe_sched));
  return isl_map_from_union_map(local);
}

/* Map the domain of "access" to the outer data->local_depth
 * schedule dimensions.   
 */
static __isl_give isl_map *local_access_io(struct polysa_array_ref_group *group,
	__isl_keep isl_union_map *access, struct polysa_group_data *data)
{
	isl_union_map *local;
  local = isl_union_map_copy(access);

  if (group->io_type == POLYSA_EXT_IO) {
    /* Group at the IO_L2 level */
    isl_union_map *new_sched = get_io_schedule_at_level(group->io_schedule, 2); 
    local = isl_union_map_apply_domain(local,
        new_sched);
  } else if (group->io_type == POLYSA_INT_IO) {
    /* Group at the IO_L1 level. */
    isl_union_map *new_sched = get_io_schedule_at_level(group->io_schedule, 1);
    local = isl_union_map_apply_domain(local,
        new_sched);
  }
  return isl_map_from_union_map(local);
}

static __isl_give isl_map *local_access_io_at_node(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group,
  __isl_keep isl_union_map *access, __isl_keep isl_schedule_node *node)
{
  isl_union_map *local, *sched;

  local = isl_union_map_copy(access);
  sched = prefix_with_equalities(node);
  sched = expand(sched, kernel->contraction);
  local = isl_union_map_apply_domain(local, sched);
  
  return isl_map_from_union_map(local);
}

/* Given an array access "access", check if for any index i there is
 * a shift a(p) and a stride g such that
 *
 *	a(p) + i = 0 mod g
 *
 * If so, record the information in tile->bound[i]->stride and
 * tile->bound[i]->shift.
 * Otherwise, set tile->bound[i]->stride to 1 (and tile->bound[i]->shift to 0).
 * Return isl_bool_true if any non-trivial stride was found.
 *
 * Note that the stride info returned by isl_map_get_range_stride_info
 * is of the form
 *
 *	i = o(p) + g n
 *
 * a(p) can therefore be taken to be equal to -o(p).
 */
static isl_bool detect_strides(struct polysa_array_tile *tile,
	__isl_keep isl_map *access)
{
	int i;
	isl_bool has_strides = isl_bool_false;

	for (i = 0; i < tile->n; ++i) {
		struct polysa_array_bound *bound = &tile->bound[i];
		isl_stride_info *si;

		si = isl_map_get_range_stride_info(access, i);
		bound->stride = isl_stride_info_get_stride(si);
		bound->shift = isl_aff_neg(isl_stride_info_get_offset(si));
		isl_stride_info_free(si);

		if (!has_strides)
			has_strides = isl_val_gt_si(bound->stride, 1);
		if (has_strides < 0)
			return isl_bool_error;
	}

	return has_strides;
}

/* Given an array access "access", remove the strides based
 * on the information in tile->bound[i]->stride and tile->bound[i]->shift.
 *
 * In particular let the access be A[a] and
 * let the shifts s_i(p) and the strides g_i be such that
 *
 *  S(p) + a = 0 mod G
 *
 * Replace the access by
 *
 *  A[(a + S(p))/G]
 *
 * First collect the shifts s_i into an isl_multi_aff and
 * the strides into the scaling function A[i] -> A[G i].
 * Then add the shifts to the original access and
 * take the preimage over the scaling.
 */
static __isl_give isl_map *remove_strides(__isl_take isl_map *access,
	struct polysa_array_tile *tile)
{
	int i;
	isl_space *space;
	isl_multi_aff *shift, *scale;
	isl_multi_val *stride;

	space = isl_map_get_space(access);
	shift = isl_multi_aff_zero(isl_space_copy(space));
	space = isl_space_range(space);
	stride = isl_multi_val_zero(isl_space_copy(space));
	scale = isl_multi_aff_identity(isl_space_map_from_set(space));
	for (i = 0; i < tile->n; ++i) {
		struct polysa_array_bound *bound = &tile->bound[i];
		isl_aff *shift_i;
		isl_val *stride_i;

		shift_i = isl_aff_copy(bound->shift);
		stride_i = isl_val_copy(bound->stride);
		shift = isl_multi_aff_set_aff(shift, i, shift_i);
		stride = isl_multi_val_set_val(stride, i, stride_i);
	}
	scale = isl_multi_aff_scale_multi_val(scale, stride);

	access = isl_map_sum(access, isl_map_from_multi_aff(shift));
	access = isl_map_preimage_range_multi_aff(access, scale);

	return access;
}

/* Check if we can find a memory tile for the given array
 * based on the given accesses, and if so, put the results in "tile".
 *
 * We project the accesses on each index in turn and look for a parametric
 * offset such that the size is constant, after removing
 * any stride that may appear in the accesses.
 *
 * tile->depth is initialized to the input dimension of the computed bounds.
 */
isl_bool can_tile(__isl_keep isl_map *access,
	struct polysa_array_tile *tile)
{
	int i;
	isl_bool has_strides, valid;
	isl_fixed_box *box;
	isl_multi_aff *offset;
	isl_multi_val *size;

	if (!tile)
		return isl_bool_error;

	isl_map_free(isl_map_detect_equalities(isl_map_copy(access)));

	has_strides = detect_strides(tile, access);
	if (has_strides < 0)
		return isl_bool_error;

	tile->depth = isl_map_dim(access, isl_dim_in);

	access = isl_map_copy(access);
	if (has_strides)
		access = remove_strides(access, tile);

	box = isl_map_get_range_simple_fixed_box_hull(access);
	isl_map_free(access);

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_map_get_ctx(access), stdout);
//  // debug

	valid = isl_fixed_box_is_valid(box);
	if (valid >= 0 && valid) {
		offset = isl_fixed_box_get_offset(box);
		size = isl_fixed_box_get_size(box);
		for (i = 0; i < tile->n; ++i) {
			tile->bound[i].size = isl_multi_val_get_val(size, i);
			tile->bound[i].lb = isl_multi_aff_get_aff(offset, i);
//      // debug
//      p = isl_printer_print_val(p, tile->bound[i].size);
//      printf("\n");
//      p = isl_printer_print_aff(p, tile->bound[i].lb);
//      printf("\n");
//      // debug
		}
		isl_multi_aff_free(offset);
		isl_multi_val_free(size);
	}
	isl_fixed_box_free(box);

	return valid;
}

struct compute_local_tile_acc_data {
  struct polysa_kernel *kernel;
  struct polysa_array_ref_group *group;
  int depth;
  isl_union_map *prefix;
  isl_union_pw_multi_aff *prefix_upma;
  int status;
};

static isl_bool compute_local_tile_acc_single(__isl_keep isl_set *set, void *user)
{
  isl_space *space;
  isl_id *id;
  struct polysa_stmt *stmt;
  struct compute_local_tile_acc_data *data = user;
  struct polysa_stmt_access *accesses, *access;

  space = isl_set_get_space(set);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  isl_space_free(space);
  stmt = find_stmt(data->kernel->prog, id);
  isl_id_free(id);
  accesses = stmt->accesses;

  for (access = accesses; access; access = access->next) {
    if (access == data->group->refs[0]) {
      return isl_bool_false;  
    }
  }

  return isl_bool_true;
}

static __isl_give isl_schedule_node *compute_local_tile_acc(__isl_take isl_schedule_node *node, void *user)
{
  struct compute_local_tile_acc_data *data = user;
  struct polysa_array_ref_group *group = data->group;
  struct polysa_stmt_access *acc = group->refs[0];
  isl_union_set *domain;
  isl_union_map *prefix;
  isl_union_pw_multi_aff *prefix_upma;
  isl_bool not_contain_acc;
  int depth;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  domain = isl_schedule_node_get_domain(node);
  not_contain_acc = isl_union_set_every_set(domain, &compute_local_tile_acc_single, data);
  isl_union_set_free(domain);

  if (!not_contain_acc) {
    int is_simd;
    is_simd = is_node_under_simd(node);
    if (is_simd) {
      isl_schedule_node *new_node;

      new_node = isl_schedule_node_copy(node);
      new_node = polysa_tree_move_up_to_mark(new_node, "simd");
      prefix = isl_schedule_node_get_prefix_schedule_union_map(new_node);
      prefix_upma = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(new_node);
      depth = isl_schedule_node_get_schedule_depth(new_node);
      isl_schedule_node_free(new_node);
    } else {
      prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
      prefix_upma = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
      depth = isl_schedule_node_get_schedule_depth(node);
    }
    if (data->depth == -1) {
      data->depth = depth;
      data->prefix = prefix;
      data->prefix_upma = prefix_upma;
      data->status = 1;
    } else {
      /* The array reference is found in more than one loop. We will compute the tiling at the PE level. */
      isl_union_map_free(prefix);
      isl_union_pw_multi_aff_free(prefix_upma);
      data->status = 0; 
    }
  }

  return node;
}

static isl_stat compute_group_bounds_core_pe_acc(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;
  isl_schedule_node *node;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write) 
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;

  /* Collect all accesses in the group */
  access = polysa_array_ref_group_access_relation(group, 1, 1);
  /* Create local tile */
  if (use_local) {
    struct compute_local_tile_acc_data tile_data;
  
    tile_data.kernel = kernel;
    tile_data.group = group;
    tile_data.status = 0;
    tile_data.depth = -1;
    tile_data.prefix = NULL;
    /* Create a tile. */
    group->local_tile = polysa_array_tile_create(ctx, group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    node = isl_schedule_get_root(kernel->schedule);
    node = polysa_tree_move_down_to_pe(node, kernel->core);
    node = isl_schedule_node_map_descendant_bottom_up(node, &compute_local_tile_acc, &tile_data);
    isl_schedule_node_free(node);
    if (tile_data.status) {
      acc = isl_map_from_union_map(isl_union_map_apply_domain(isl_union_map_copy(access),
          tile_data.prefix));
      /* Update the copy schedule */
      group->copy_schedule_dim = tile_data.depth;
      group->copy_schedule = tile_data.prefix_upma;
      group->copy_schedule = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(group->copy_schedule, 
        isl_union_pw_multi_aff_copy(kernel->contraction));
    } else {
      acc = local_access_pe(group, access, data);
      /* Update the copy schedule */
      node = isl_schedule_get_root(kernel->schedule);
      node = polysa_tree_move_down_to_pe(node, kernel->core);
      group->copy_schedule_dim = isl_schedule_node_get_schedule_depth(node);
      group->copy_schedule = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
      group->copy_schedule = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(group->copy_schedule, 
          isl_union_pw_multi_aff_copy(kernel->contraction));
      isl_schedule_node_free(node);
    }
    /* Collect the shift and scale factors of the tile. */
    ok = can_tile(acc, group->local_tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      group->local_tile = polysa_array_tile_free(group->local_tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

/* If the any reference in the group is associated with RAW/RAW not carried at space
 * loop (contains interior I/O), compute the local memory tiles for the group.
 * Return isl_stat_ok on success and isl_stat_error on error.
 *
 * If the array is a read-only scalar or if the user requested not to use 
 * local emory, then we do not need to do anything.
 *
 * The tiling is computed at the PE level.
 */
static isl_stat compute_group_bounds_core_pe(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write)
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;

  /* Collect all accesses in the group. */
  access = polysa_array_ref_group_access_relation(group, 1, 1); 
  /* Create local tile */
  if (use_local) {
    /* Create a tile. */
    group->local_tile = polysa_array_tile_create(ctx,
            group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    // TODO: to create a register tiling or tile under the SIMD loop
    acc = local_access_pe(group, access, data); 
    /* Collect the shift and scale factors of the tile. */
    ok = can_tile(acc, group->local_tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      group->local_tile = 
        polysa_array_tile_free(group->local_tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

/* Compute the local memory tiles for the array reference group "group"
 * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
 *
 * If the array is a read-only scalar or if the user requested not to use local
 * memory, then we do not need to do anything.
 */
isl_stat compute_group_bounds_io_at_node(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, __isl_keep isl_schedule_node *node,
  struct polysa_io_buffer *buffer) 
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write)
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;
 
  /* Collect all accesses in the group. */
  access = polysa_array_ref_group_access_relation(group, 1, 1);
  /* Create local tile */
  if (use_local) {
    /* Create a tile */
    buffer->tile = polysa_array_tile_create(ctx, group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    acc = local_access_io_at_node(kernel, group, access, node);
    /* Collect the shift and scale factors of the tile */
    ok = can_tile(acc, buffer->tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      buffer->tile = polysa_array_tile_free(buffer->tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

/* Compute the local memory tiles for the array reference group "group"
 * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
 *
 * If the array is a read-only scalar or if the user requested not to use local
 * memory, then we do not need to do anything.
 */
isl_stat compute_group_bounds_drain_at_node(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, __isl_keep isl_schedule_node *node,
  struct polysa_io_buffer *buffer) 
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write)
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;
 
  /* Collect all accesses in the group. */
  access = polysa_array_ref_group_access_relation(group, 0, 1);
  /* Create local tile */
  if (use_local) {
    /* Create a tile */
    buffer->tile = polysa_array_tile_create(ctx, group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    acc = local_access_io_at_node(kernel, group, access, node);
    /* Collect the shift and scale factors of the tile */
    ok = can_tile(acc, buffer->tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      buffer->tile = polysa_array_tile_free(buffer->tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

/* Compute the local memory tiles for the array reference group "group"
 * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
 *
 * If the array is a read-only scalar or if the user requested not to use 
 * local emory, then we do not need to do anything.
 *
 * For interior I/O group, the tiling is computed at the PE level.
 * For exteriro I/O group, the tiling is computed at the PE ray level.
 */
static isl_stat compute_group_bounds_core_io(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write)
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;

  /* Collect all accesses in the group. 
   * TODO: Overapproximation */
  access = polysa_array_ref_group_access_relation(group, 1, 1); 
  /* Create local tile */
  if (use_local) {
    /* Create a tile. */
    group->local_tile = polysa_array_tile_create(ctx,
            group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    acc = local_access_io(group, access, data); 
    /* Collect the shift and scale factors of the tile. */
    ok = can_tile(acc, group->local_tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      group->local_tile = 
        polysa_array_tile_free(group->local_tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

/* Compute the local memory tiles for the array reference group "group"
 * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
 *
 * The tiling is computed at the PE level.
 */
static isl_stat compute_group_bounds_core_drain(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
  isl_ctx *ctx = isl_space_get_ctx(group->array->space);
  int use_local = kernel->options->use_local_memory;
  isl_stat r = isl_stat_ok;
  isl_union_map *access;
  isl_map *acc;
  isl_bool ok;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  if (!group->exact_write)
    return isl_stat_ok;
  if (group->slice)
    return isl_stat_ok;

  /* Collect all accesses in the group. */
  /* This is overapproximated. */
  access = polysa_array_ref_group_access_relation(group, 0, 1); 
  /* Create local tile */
  if (use_local) {
    /* Create a tile. */
    group->local_tile = polysa_array_tile_create(ctx,
            group->array->n_index);
    /* Map the domain to the outer scheduling dimensions */
    acc = local_access_io(group, access, data); 
    /* Collect the shift and scale factors of the tile. */
    ok = can_tile(acc, group->local_tile);
    if (ok < 0)
      r = isl_stat_error;
    else if (!ok)
      group->local_tile = 
        polysa_array_tile_free(group->local_tile);
    isl_map_free(acc);
  }

  if (r < 0) {
    isl_union_map_free(access);
    return r;
  }

  isl_union_map_free(access);
  return isl_stat_ok;
}

///* Compute the local memory tiles for the array reference group "group" 
// * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
// *
// * If the array is a read-only scalar or if the user requested not to use
// * local memory, then we do not need to do anything.
// *
// * For FPGA, we will compute a local memory tile for every reference group.
// */
//static isl_stat compute_group_bounds_core(struct polysa_kernel *kernel,
//	struct polysa_array_ref_group *group, struct polysa_group_data *data)
//{
//	isl_ctx *ctx = isl_space_get_ctx(group->array->space);
//	isl_union_map *access, *local;
//	int n_index = group->array->n_index;
//	int no_reuse, coalesced;
//	isl_map *acc;
//	// int force_private = group->local_array->force_private;
//	// int use_shared = !force_private && kernel->options->use_shared_memory &&
//	// 			data->n_thread > 0;
//	// int use_private = force_private || kernel->options->use_private_memory;
//  int use_local = kernel->options->use_local_memory;
//	isl_stat r = isl_stat_ok;
//	isl_bool ok;
//	int requires_unroll;
//	int unique_depth;
//
//  if (!use_local)
//    return isl_stat_ok;
//  if (polysa_array_is_read_only_scalar(group->array))
//    return isl_stat_ok;
//  /* PPCG: If the array group involves any may writes (that are not must writes),
//   * then we would have to make sure that we load the data into local memory first
//   * in case the data is not written by the kernel (but still written back out to global
//   * memory).
//   * Since we don't have any such mechanism at the moment, we don't compute 
//   * local tiles for groups involving may writes.
//   */
//  if (!group->exact_write)
//    return isl_stat_ok;
//  /* PPCG: If any reference in the reference group accesses more than one element,
//   * then we would have to make sure that the layout in shared memory is the same as that
//   * in global memory. Since we do not handle this yet (and it may not even be possible),
//   * we refuse to map to local memory in such cases.
//   */
//  if (group->slice)
//    return isl_stat_ok;
//
//	access = polysa_array_ref_group_access_relation(group, 1, 1); 
//	local = localize_access(data, isl_union_map_copy(access));
//	no_reuse = isl_union_map_is_injective(local);
//	//if (no_reuse < 0)
//	//	r = isl_stat_error;
//	//if (use_shared && no_reuse)
//	//	coalesced = access_is_coalesced(data, local);
//  isl_union_map_free(local);
//
//	//if (r >= 0 && kernel->options->debug->verbose &&
//	//    use_shared && no_reuse && coalesced)
//	//	report_no_reuse_and_coalesced(kernel, access);
//
//  /* Create local tile */
//  if (use_local) {
//    /* Create a tile. */
//    group->local_tile = polysa_array_tile_create(ctx,
//            group->array->n_index);
//    acc = local_access(group, access, data);
//    /* Collect the shift and scale factors of the tile. */
//    ok = can_tile(acc, group->local_tile);
//    if (ok < 0)
//      r = isl_stat_error;
//    else if (!ok)
//      group->local_tile =
//        polysa_array_tile_free(group->local_tile);
//    isl_map_free(acc);
//  }
//
//  if (r < 0) {
//    isl_union_map_free(access);
//    return r;
//  }
//    
//  /* Calculation for private tile, commented out */
//  /* PPCG: 
//   * For computing a private memory tile, we also require that there is
//   * some reuse.  Moreover, we require that the access is private
//   * to the thread.  That is, we check that any given array element
//   * is only accessed by a single thread.
//   * We compute an access relation that maps the outer
//   * data->thread_depth + data->n_thread schedule dimensions.
//   * The latter data->n_thread will be mapped to thread identifiers.
//   * We actually check that those iterators that will be wrapped
//   * partition the array space.  This check is stricter than necessary
//   * since several iterations may be mapped onto the same thread
//   * and then they could be allowed to access the same memory elements,
//   * but our check does not allow this situation.
//   *
//   * For private memory tiles, the number of schedule dimensions that
//   * affect the offset is computed and stored in tile->depth, with
//   * a lower bound of data->kernel_depth.  If this depth is smaller
//   * than the minimal depth that still ensures that every element
//   * is accessed by a single thread, then the depth is raised
//   * to this minimal depth.
//   * The fields of the tile are then adjusted to only refer to the tile->depth
//   * outer schedule dimensions.
//   *
//   * We also check that the index expression only depends on parallel
//   * loops.  That way, we can move those loops innermost and unroll them.
//   * Again, we use a test that is stricter than necessary.
//   * We actually check whether the index expression only depends
//   * on the iterators that are wrapped over the threads.
//   * These are necessarily parallel, but there may be more parallel loops.
//   *
//   * Combining the injectivity of the first test with the single-valuedness
//   * of the second test, we simply test for bijectivity.
//   *
//   * If the use of the private tile requires unrolling, but some
//   * of the other arrays are forcibly mapped to private memory,
//   * then we do not allow the use of this private tile since
//   * we cannot move the schedule dimensions that need to be unrolled down
//   * without performing some kind of expansion on those arrays
//   * that are forcibly mapped to private memory.
//   *
//   * If the array is marked force_private, then we bypass all checks
//   * and assume we can (and should) use registers only.
//   *
//   * If it turns out we can (or have to) use registers, we compute
//   * the private memory tile size using can_tile, after introducing a dependence
//   * on the thread indices.
//   */
//
////	access = isl_union_map_apply_domain(access,
////					isl_union_map_copy(data->thread_sched));
////
////	acc = isl_map_from_union_map(access);
////
////	if (!force_private && !access_is_bijective(data, acc)) {
////		isl_map_free(acc);
////		return isl_stat_ok;
////	}
////
////	unique_depth = compute_accessed_by_single_thread_depth(data, acc);
////
////	acc = isl_map_intersect_domain(acc, isl_set_copy(data->privatization));
////	acc = isl_map_project_out(acc, isl_dim_in, data->thread_depth,
////								data->n_thread);
////	requires_unroll = check_requires_unroll(data, acc, force_private);
////	if (unique_depth < 0 || requires_unroll < 0 ||
////	    (requires_unroll && kernel->any_force_private)) {
////		isl_map_free(acc);
////		return requires_unroll < 0 ? isl_stat_error : isl_stat_ok;
////	}
////
////	group->private_tile = gpu_array_tile_create(ctx, n_index);
////	group->private_tile->requires_unroll = requires_unroll;
////	ok = can_tile(acc, group->private_tile);
////	if (ok >= 0 && !ok)
////		group->private_tile = gpu_array_tile_free(group->private_tile);
////	isl_map_free(acc);
////	if (ok < 0)
////		return isl_stat_error;
////
////	if (group->private_tile) {
////		struct gpu_array_tile *tile = group->private_tile;
////		int tile_depth = compute_tile_depth(data, tile);
////		if (tile_depth < unique_depth)
////			tile_depth = unique_depth;
////		if (tile_adjust_depth(tile, tile_depth) < 0)
////			return isl_stat_error;
////	}
////
////	if (force_private && !group->private_tile)
////		isl_die(ctx, isl_error_internal,
////			"unable to map array reference group to registers",
////			return isl_stat_error);
//
//	return isl_stat_ok;
//}

/* Compute the number of outer schedule tile dimensions that affect
 * the offset of "tile".
 * If there is no such dimension, then return the index
 * of the first kernel dimension, i.e., data->kernel_depth.
 */
static int compute_tile_depth(struct polysa_group_data *data,
	struct polysa_array_tile *tile)
{
	int i, j;

	for (j = tile->depth - 1; j >= data->kernel_depth; --j) {
		for (i = 0; i < tile->n; ++i) {
			isl_aff *lb;
			isl_aff *shift;

			lb = tile->bound[i].lb;
			if (isl_aff_involves_dims(lb, isl_dim_in, j, 1))
				break;

			shift = tile->bound[i].shift;
			if (!shift)
				continue;
			if (isl_aff_involves_dims(shift, isl_dim_in, j, 1))
				break;
		}
		if (i < tile->n)
			break;
	}

	return ++j;
}

/* Adjust the fields of "tile" to reflect the new input dimension "depth".
 * The dimension beyond "depth" are assumed not to affect the tile,
 * so they can simply be dropped.
 */
static int tile_adjust_depth(struct polysa_array_tile *tile, int depth)
{
	int i;

	if (tile->depth == depth)
		return 0;

	for (i = 0; i < tile->n; ++i) {
		tile->bound[i].lb = isl_aff_drop_dims(tile->bound[i].lb,
					isl_dim_in, depth, tile->depth - depth);
		if (!tile->bound[i].lb)
			return -1;
		if (!tile->bound[i].shift)
			continue;
		tile->bound[i].shift = isl_aff_drop_dims(tile->bound[i].shift,
					isl_dim_in, depth, tile->depth - depth);
		if (!tile->bound[i].shift)
			return -1;
	}

	tile->depth = depth;

	return 0;
}

/* Determine the number of schedule dimensions that affect the offset of the
 * local tile "tile" and store the result in tile->depth, with
 * a lower bound of data->kernel_depth.
 * Also adjust the fields of the tile to only refer to the tile->depth
 * outer schedule dimensions.
 */
static isl_stat tile_set_depth(struct polysa_group_data *data,
	struct polysa_array_tile *tile)
{
	if (tile_adjust_depth(tile, compute_tile_depth(data, tile)) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

///* Determine the number of schedule dimensions that affect the offset of the
// * local tile and store it in group->min_depth, with a lower bound of data->kernel_depth.
// * If there is no tile defined on the array reference group,
// * then set group->min_depth to data->local_depth.
// */
//static int set_depth(struct polysa_group_data *data,
//	struct polysa_array_ref_group *group)
//{
//	group->min_depth = data->local_depth;
//
//	if (group->local_tile) {
//		if (tile_set_depth(data, group->local_tile) < 0) 
//			return -1;
//		if (group->local_tile->depth < group->min_depth)
//			group->min_depth = group->local_tile->depth;
//	}
//
//	return 0;
//}

/* Compute the local memory tiles for the array
 * reference group "group" of array "array" and set the tile depth.
 * Return 0 on success and -1 on error.
 */
static int compute_group_bounds_pe(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
	if (!group)
		return -1;
	if (compute_group_bounds_core_pe(kernel, group, data) < 0) 
		return -1;

	return 0;
}

static int compute_group_bounds_pe_acc(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
  if (!group)
    return -1;
  if (compute_group_bounds_core_pe_acc(kernel, group, data) < 0)
    return -1;

  return 0;
}

/* Compute the local memory tiles for the array
 * reference group "group" of array "array" and set the tile depth.
 * Return 0 on success and -1 on error.
 */
static int compute_group_bounds_io(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
	if (!group)
		return -1;
	if (compute_group_bounds_core_io(kernel, group, data) < 0) 
		return -1;

	return 0;
}

/* Compute the local memory tiles for the array
 * reference group "group" of array "array" and set the tile depth.
 * Return 0 on success and -1 on error.
 */
static int compute_group_bounds_drain(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
	if (!group)
		return -1;
	if (compute_group_bounds_core_drain(kernel, group, data) < 0) 
		return -1;

	return 0;
}


/* If two groups have shared I/O (as determined by
 * the "share" function),
 * then merge the two groups into one.
 * TODO: If "compute_bounds" is set, then call compute_group_bounds
 * on the merged groups.
 *
 * Return the updated number of groups.
 * Return -1 on error.
 */
static int group_io(struct polysa_kernel *kernel,
	int n, struct polysa_array_ref_group **groups,
	int (*share)(struct polysa_array_ref_group *group1,
		struct polysa_array_ref_group *group2), int compute_bounds,
	struct polysa_group_data *data)
{
	int i, j;

	for (i = 0; i < n; ++i) {
		for (j = n - 1; j > i; --j) {
			if (!share(groups[i], groups[j]))
				continue;

			groups[i] = join_groups_and_free(groups[i], groups[j]); 
			if (j != n - 1)
				groups[j] = groups[n - 1];
			groups[n - 1] = NULL;
			n--;

			if (!groups[i])
				return -1;
//			if (compute_bounds &&
//			    compute_group_bounds_io(kernel, groups[i], data) < 0) 
//				return -1;
		}
	}

	return n;
}

///* If two groups have overlapping access relations (as determined by
// * the "overlap" function) and if one of them involves a write,
// * then merge the two groups into one.
// * If "compute_bounds" is set, then call compute_group_bounds
// * on the merged groups.
// * If any group is merged into the current group, then its access
// * relation may have changed or it may have been turned into a write.
// * The combined group might therefore overlap with groups that
// * the original group did not overlap with. The groups therefore
// * need to be checked again.
// *
// * Return the updated number of groups.
// * Return -1 on error.
// */
//static int group_writes(struct polysa_kernel *kernel,
//	int n, struct polysa_array_ref_group **groups,
//	int (*overlap)(struct polysa_array_ref_group *group1,
//		struct polysa_array_ref_group *group2), int compute_bounds,
//	struct polysa_group_data *data)
//{
//	int i, j;
//	int any_merge;
//
//	for (i = 0; i < n; i += !any_merge) {
//		any_merge = 0;
//		for (j = n - 1; j > i; --j) {      
//			if (!groups[i]->write && !groups[j]->write)
//				continue;
//
//			if (!overlap(groups[i], groups[j]))
//				continue;
//
//			any_merge = 1;
//			groups[i] = join_groups_and_free(groups[i], groups[j]); 
//			if (j != n - 1)
//				groups[j] = groups[n - 1];
//			groups[n - 1] = NULL;
//			n--;
//
//			if (!groups[i])
//				return -1;
//			if (compute_bounds &&
//			    compute_group_bounds(kernel, groups[i], data) < 0) 
//				return -1;
//		}
//	}
//
//	return n;
//}

/* Check if the access relations of group1 and group2 overlap within
 * copy_sched.
 */
static int accesses_overlap(struct polysa_array_ref_group *group1,
	struct polysa_array_ref_group *group2)
{
	int disjoint;

	disjoint = isl_map_is_disjoint(group1->access, group2->access);
	if (disjoint < 0)
		return -1;

	return !disjoint;
}

static int share_io(struct polysa_array_ref_group *group1,
	struct polysa_array_ref_group *group2)
{
  if (group1->io_type != group2->io_type)
    return 0;

  for (int i = 0; i < isl_vec_size(group1->dir); i++) {
    if (isl_vec_cmp_element(group1->dir, group2->dir, i))
      return 0;
  }

  return 1;
}

/* If two groups share the same I/O type and data transfer direction,
 * then merge the two groups into one.
 *
 * Return the updated number of groups.
 */
static int group_share_io(struct polysa_kernel *kernel,
	int n, struct polysa_array_ref_group **groups,
	struct polysa_group_data *data)
{
	return group_io(kernel, n, groups, &share_io, 0, data);
}

///* If two groups have overlapping access relations (within the innermost
// * loop) and if one of them involves a write, then merge the two groups
// * into one.
// *
// * Return the updated number of groups.
// */
//static int group_overlapping_writes(struct polysa_kernel *kernel,
//	int n, struct polysa_array_ref_group **groups,
//	struct polysa_group_data *data)
//{
//	return group_writes(kernel, n, groups, &accesses_overlap, 0, data);
//}

/* Check if the access relations of group1 and group2 overlap within
 * the outermost min(group1->min_depth, group2->min_depth) loops.
 */
static int depth_accesses_overlap(struct polysa_array_ref_group *group1,
	struct polysa_array_ref_group *group2)
{
	int depth;
	int dim;
	int empty;
	isl_map *map_i, *map_j, *map;

	depth = group1->min_depth;
	if (group2->min_depth < depth)
		depth = group2->min_depth;
	map_i = isl_map_copy(group1->access);
	dim = isl_map_dim(map_i, isl_dim_in);
	map_i = isl_map_eliminate(map_i, isl_dim_in, depth, dim - depth);
	map_j = isl_map_copy(group2->access);
	map_j = isl_map_eliminate(map_j, isl_dim_in, depth, dim - depth);
	map = isl_map_intersect(map_i, map_j);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	return !empty;
}

///* If two groups have overlapping access relations (within the outer
// * depth loops) and if one of them involves a write,
// * then merge the two groups into one.
// *
// * Return the updated number of groups.
// */
//static int group_depth_overlapping_writes(struct polysa_kernel *kernel,
//	int n, struct polysa_array_ref_group **groups, struct polysa_group_data *data)
//{
//	return group_writes(kernel, n, groups, &depth_accesses_overlap, 1,
//				data);
//}

/* Is the size of the tile specified by "tile" smaller than the sum of
 * the sizes of the tiles specified by "tile1" and "tile2"?
 */
static int smaller_tile(struct polysa_array_tile *tile,
	struct polysa_array_tile *tile1, struct polysa_array_tile *tile2)
{
	int smaller;
	isl_val *size, *size1, *size2;

	size = polysa_array_tile_size(tile);
	size1 = polysa_array_tile_size(tile1);
	size2 = polysa_array_tile_size(tile2);

	size = isl_val_sub(size, size1);
	size = isl_val_sub(size, size2);
	smaller = isl_val_is_neg(size);

	isl_val_free(size);

	return smaller;
}

///* Given an initial grouping of array references and local memory tiles
// * for each group that allows for a local memory tile, merge two groups
// * if both have a local memory tile, the merged group also has
// * a local memory tile and the size of the tile for the merge group
// * is smaller than the sum of the tile sizes of the individual groups.
// * If any group is merged into the current group, then it may become
// * profitable to combine it with groups that were considered before
// * the merge.  The groups are therefore checked again after a merge.
// *
// * If merging two groups decreases the depth of the tile of
// * one or both of the two groups, then we need to check for overlapping
// * writes again.
// *
// * Return the number of groups after merging.
// * Return -1 on error.
// */
//static int group_common_local_memory_tile(struct polysa_kernel *kernel,
//	struct polysa_array_info *array, int n,
//	struct polysa_array_ref_group **groups, struct polysa_group_data *data)
//{
//	int i, j;
//	int recompute_overlap = 0;
//	int any_merge;
//
//	for (i = 0; i < n; i += !any_merge) {
//		any_merge = 0;
//		if (!groups[i]->local_tile)
//			continue;
//		for (j = n - 1; j > i; --j) {
//			struct polysa_array_ref_group *group;
//
//			if (!groups[j]->local_tile)
//				continue;
//
//			if (!depth_accesses_overlap(groups[i], groups[j]))
//				continue;
//
//			group = join_groups(groups[i], groups[j]);
//			if (compute_group_bounds(kernel, group, data) < 0) {
//				polysa_array_ref_group_free(group);
//				return -1;
//			}
//			if (!group->local_tile ||
//			    !smaller_tile(group->local_tile,
//					groups[i]->local_tile,
//					groups[j]->local_tile)) {
//				polysa_array_ref_group_free(group);
//				continue;
//			}
//
//			any_merge = 1;
//			if (group->min_depth < groups[i]->min_depth ||
//			    group->min_depth < groups[j]->min_depth)
//				recompute_overlap = 1;
//			polysa_array_ref_group_free(groups[i]);
//			polysa_array_ref_group_free(groups[j]);
//			groups[i] = group;
//			if (j != n - 1)
//				groups[j] = groups[n - 1];
//			n--;
//		}
//	}
//
//	if (recompute_overlap)
//		n = group_depth_overlapping_writes(kernel, n, groups, data);
//	return n;
//}

/* Set array->n_group and array->groups to n and groups.
 *
 * Additionally, set the "nr" field of each group.
 */
static void set_array_groups_pe(struct polysa_local_array_info *array,
	int n, struct polysa_array_ref_group **groups)
{
	int i;

	array->n_pe_group = n;
	array->pe_groups = groups;

	for (i = 0; i < n; ++i)
	  groups[i]->nr = i;
}

/* Set array->n_group and array->groups to n and groups.
 *
 * Additionally, set the "nr" field of each group.
 */
static void set_array_groups_io(struct polysa_local_array_info *array,
	int n, struct polysa_array_ref_group **groups)
{
	int i;

	array->n_io_group = n;
	array->io_groups = groups;

	for (i = 0; i < n; ++i)
		groups[i]->nr = i;
}

/* Populate the array reference groups with single array reference.
 * If any of the array reference is associated with RAW, the array reference
 * is from an internal array, we will merge all the array references into 
 * one single group.
 * Otherwise, the array reference is from an external array, we do nothing
 * here.
 * For internal array, we compute the group tiling at the PE level.
 * For external array, registers are allocated for each access. 
 * Set the group tiling as NULL.
 * Return -1 on error.
 */
static int group_array_references_pe(struct polysa_kernel *kernel,
  struct polysa_local_array_info *local, struct polysa_group_data *data)
{
  int i, j;
  int n;
  isl_ctx *ctx = isl_union_map_get_ctx(data->pe_sched);
  struct polysa_array_ref_group **groups;
  int merge_all = 0;
  isl_schedule_node *node;

  groups = isl_calloc_array(ctx, struct polysa_array_ref_group *, 
      local->array->n_ref);
  if (!groups)
    return -1;

  n = populate_array_references_pe(local, groups, data);

  /* Examine if any of the array references is associated with RAW or
   * RAR carried at space loop. If then, merge all the groups. 
   */
  for (int i = 0; i < n; ++i) {
    struct polysa_array_ref_group *group_i = groups[i];
    for (int j = 0; j < group_i->n_ref; ++j) {
      struct polysa_stmt_access *ref_i = group_i->refs[j];
      for (int k = 0; k < ref_i->n_io_info; ++k) {
        if (ref_i->io_info[k]->dep->type == POLYSA_DEP_RAW) {
          merge_all = 1;
          break;
        }
      }
    }
  }

  if (merge_all) {
    /* Join all referneces together */
    for (int i = 1; i < n; ++i) {
      groups[0] = join_groups_and_free(groups[0], groups[i]);
    }
    n = 1;
  }

  if (merge_all) {
    /* Internal array */
    for (i = 0; i < n; ++i) { 
      if (compute_group_bounds_pe(kernel, groups[i], data) < 0) {
        for (j = 0; j < n; j++) {
          polysa_array_ref_group_free(groups[j]);
        }
        free(groups);
        return -1;
      }
      /* Update the copy schedule at the PE level */
      node = isl_schedule_get_root(kernel->schedule);
      node = polysa_tree_move_down_to_pe(node, kernel->core);
      groups[i]->copy_schedule_dim = isl_schedule_node_get_schedule_depth(node);
      groups[i]->copy_schedule = 
        isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
      groups[i]->copy_schedule = 
        isl_union_pw_multi_aff_pullback_union_pw_multi_aff(groups[i]->copy_schedule, 
            isl_union_pw_multi_aff_copy(kernel->contraction));
      isl_schedule_node_free(node);
    }
  } else {
    /* External array. 
     * We will build the tiling for each array access */
    for (i = 0; i < n; ++i) {
      if (compute_group_bounds_pe_acc(kernel, groups[i], data) < 0) {
        for (j = 0; j < n; j++) {
          polysa_array_ref_group_free(groups[j]);
        }
        free(groups);    
        return -1;
      }
    }
  }

  set_array_groups_pe(local, n, groups);

  return 0;
}

static __isl_give isl_schedule_node *io_cluster(__isl_take isl_schedule_node *node,
  __isl_keep isl_vec *dir, isl_mat **io_trans_mat, isl_multi_aff **io_trans_ma)
{
  isl_multi_union_pw_aff *mupa;
  isl_mat *trans_mat, *d_mat, *null_mat;
  int space_dim;
  isl_ctx *ctx;
  isl_space *space;
  isl_multi_aff *ma;

  mupa = isl_schedule_node_band_get_partial_schedule(node);
  space_dim = isl_schedule_node_band_n_member(node);
  ctx = isl_schedule_node_get_ctx(node);
  
  /* Build the transformation matrix */
  trans_mat = isl_mat_alloc(ctx, space_dim, space_dim);
  d_mat = isl_mat_alloc(ctx, 1, space_dim);
  for (int i = 0; i < isl_vec_size(dir); i++) {
    d_mat = isl_mat_set_element_val(d_mat, 0, i, 
        isl_vec_get_element_val(dir, i));
  }
  null_mat = isl_mat_right_kernel(d_mat);
  for (int i = 0; i < isl_mat_cols(null_mat); i++)
    for (int j = 0; j < isl_mat_rows(null_mat); j++) {
      trans_mat = isl_mat_set_element_val(trans_mat, i, j,
          isl_mat_get_element_val(null_mat, j, i));
    }
  for (int i = 0; i < isl_vec_size(dir); i++) {
    trans_mat = isl_mat_set_element_val(trans_mat, isl_mat_cols(null_mat), i,
          isl_vec_get_element_val(dir, i));
  }
  *io_trans_mat = trans_mat;

  /* Convert the transformation matrix to multi_aff */
  space = isl_multi_union_pw_aff_get_space(mupa);
  space = isl_space_map_from_set(space);
  ma = isl_multi_aff_identity(space);

  for (int i = 0; i < isl_mat_rows(trans_mat); i++) {
    isl_aff *aff = isl_multi_aff_get_aff(ma, i);
    for (int j = 0; j < isl_mat_cols(trans_mat); j++) {
      isl_val *val = isl_mat_get_element_val(trans_mat, i, j);
      aff = isl_aff_set_coefficient_si(aff, isl_dim_in, j, isl_val_get_num_si(val));
      isl_val_free(val);
    }
    ma = isl_multi_aff_set_aff(ma, i, aff);
  }

  mupa = isl_multi_union_pw_aff_apply_multi_aff(mupa, isl_multi_aff_copy(ma));
  *io_trans_ma = ma;

  node = isl_schedule_node_delete(node);
  node = isl_schedule_node_insert_partial_schedule(node, mupa);

  isl_mat_free(null_mat);

  return node;
}

/* This function computes the schedule for the I/O modules that transfers
 * the data for the I/O group "group".
 */
static isl_stat compute_io_group_schedule(
  struct polysa_kernel *kernel, struct polysa_array_ref_group *group, 
  struct polysa_gen *gen)
{
  isl_printer *p_str;
  char *io_str;
  int io_level = 0;
  int i;
  isl_ctx *ctx = gen->ctx;
  isl_id *id;
  isl_schedule *sched;
  isl_mat *io_trans_mat = NULL;
  isl_multi_aff *io_trans_ma = NULL;
  isl_map *io_trans_map = NULL;
  isl_schedule_node *node;
  int space_dim;
  isl_schedule *schedule;

  /* Sink to the space band */
  schedule = isl_schedule_dup(kernel->schedule);
  node = isl_schedule_get_root(schedule);
  isl_schedule_free(schedule);

  node = polysa_tree_move_down_to_array(node, kernel->core);
  node = isl_schedule_node_child(node, 0);
  space_dim = isl_schedule_node_band_n_member(node);

  /* Insert the IO_L1 mark */
  node = isl_schedule_node_child(node, 0);
  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, "io_L");
  p_str = isl_printer_print_int(p_str, io_level + 1);
  io_str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  id = isl_id_alloc(ctx, io_str, NULL);
  free(io_str);
  node = isl_schedule_node_insert_mark(node, id);
  io_level++;
  node = isl_schedule_node_parent(node);

  /* Cluster the I/O modules from innermost space loops to outermost loops */
  for (int i = space_dim - 1; i >= 0; i--) {
    isl_mat *io_trans_mat_i;
    isl_multi_aff *io_trans_ma_i;
    isl_vec *dir;
    isl_mat *mat;

    /* Perform space-time transformation on the current band */
    if (i == space_dim - 1 && group->io_type == POLYSA_EXT_IO) {
      dir = isl_vec_dup(group->dir);
    } else {
      /* By default, we set the first element of the direction vector as 1 */
      dir = isl_vec_zero(ctx, i + 1);
      dir = isl_vec_set_element_si(dir, 0, 1);
    }

    node = io_cluster(node, dir, &io_trans_mat_i, &io_trans_ma_i);
    isl_vec_free(dir);

    if (io_level == 1) {
      sched = isl_schedule_node_get_schedule(node);
      group->io_L1_schedule = isl_schedule_dup(sched);
      group->io_L1_trans = isl_multi_aff_copy(io_trans_ma_i);

      isl_schedule_free(sched);
      io_trans_mat = io_trans_mat_i;
      io_trans_ma = io_trans_ma_i;
    } else {
      isl_multi_aff_free(io_trans_ma_i);
      /* Apply the transformation */
      /* Build up the transformation matrix */
      int nrow = isl_mat_rows(io_trans_mat);
      int ncol = isl_mat_cols(io_trans_mat);
      isl_mat *extend_mat = isl_mat_alloc(ctx, nrow, ncol);
      isl_mat *product_mat = isl_mat_alloc(ctx, nrow, ncol);
      for (int r = 0; r < nrow; r++)
        for (int c = 0; c < ncol; c++) {
          extend_mat = isl_mat_set_element_si(extend_mat, r, c, 0);
          product_mat = isl_mat_set_element_si(product_mat, r, c, 0);
        }

      for (int r = 0; r < isl_mat_rows(io_trans_mat_i); r++)
        for (int c = 0; c < isl_mat_cols(io_trans_mat_i); c++) {
          extend_mat = isl_mat_set_element_val(extend_mat, r, c,
              isl_mat_get_element_val(io_trans_mat_i, r, c));
        }
      for (int r = isl_mat_rows(io_trans_mat_i); r < nrow; r++) {
        extend_mat = isl_mat_set_element_si(extend_mat, r, r, 1);
      }
      for (int r = 0; r < nrow; r++)
        for (int c = 0; c < ncol; c++) {
          for (int k = 0; k < nrow; k++) {
            isl_val *v1, *v2, *v3;
            v1 = isl_mat_get_element_val(extend_mat, r, k);
            v2 = isl_mat_get_element_val(io_trans_mat, k, c);
            v3 = isl_mat_get_element_val(product_mat, r, c);
            v1 = isl_val_mul(v1, v2);
            v3 = isl_val_add(v1, v3);
            product_mat = isl_mat_set_element_val(product_mat, r, c, v3);
          }
        }
      isl_mat_free(io_trans_mat);
      isl_mat_free(extend_mat);
      isl_mat_free(io_trans_mat_i);
      io_trans_mat = product_mat;
      /* Reset the transformation function */
      for (int r = 0; r < nrow; r++) {
        isl_aff *aff = isl_multi_aff_get_aff(io_trans_ma, r);
        for (int c = 0; c < ncol; c++) {
          isl_val *val = isl_mat_get_element_val(io_trans_mat, r, c);
          aff = isl_aff_set_coefficient_si(aff, isl_dim_in, c, isl_val_get_num_si(val));
          isl_val_free(val);
        }
        io_trans_ma = isl_multi_aff_set_aff(io_trans_ma, r, aff);
      }        
    }

    /* Split the band and insert the IO mark */
    if (i > 0) {
      node = isl_schedule_node_band_split(node, i);
      node = isl_schedule_node_child(node, 0);
    }

    /* If the multi-port DRAM/HBM is to be used, we will need to tile the loop again */
    if (i == 0 && gen->options->hbm) {
      printf("[PolySA] Apply HBM optimization.\n");
      if (group->io_type == POLYSA_EXT_IO && i == space_dim - 1) { 
        printf("[PolySA] HBM optimization failed! Not enough I/O modules.\n");
        goto next; 
      }

      int tile_len = 1;
      int *tile_size = NULL;
      tile_size = read_hbm_tile_sizes(kernel, &tile_len);
      printf("[PolySA] HBM port: %d\n", tile_size[0]);
      node = polysa_tile_band(node, tile_size);
      node = isl_schedule_node_child(node, 0);
      space_dim++;

      /* Update the transformation function */
      isl_aff *aff = isl_multi_aff_get_aff(io_trans_ma, 0);
      isl_aff *tile_aff, *point_aff;
      tile_aff = isl_aff_scale_down_ui(isl_aff_copy(aff), tile_size[0]);
      tile_aff = isl_aff_floor(tile_aff);
      point_aff = isl_aff_scale_down_ui(isl_aff_copy(aff), tile_size[0]);
      point_aff = isl_aff_floor(point_aff);
      point_aff = isl_aff_scale_val(point_aff, isl_val_int_from_ui(ctx, tile_size[0]));
      point_aff = isl_aff_sub(aff, point_aff);

      isl_aff_list *aff_list = isl_aff_list_from_aff(tile_aff);
      aff_list = isl_aff_list_add(aff_list, point_aff);
      for (int n = 1; n < isl_multi_aff_dim(io_trans_ma, isl_dim_out); n++) {
        aff = isl_multi_aff_get_aff(io_trans_ma, n);
        aff_list = isl_aff_list_add(aff_list, aff);
      }

      isl_space *space = isl_multi_aff_get_space(io_trans_ma);
      isl_multi_aff_free(io_trans_ma);
      space = isl_space_add_dims(space, isl_dim_out, 1);
      io_trans_ma = isl_multi_aff_from_aff_list(space, aff_list);
      free(tile_size);
    }
next:
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, "io_L");
    p_str = isl_printer_print_int(p_str, io_level + 1);
    io_str = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    id = isl_id_alloc(ctx, io_str, NULL);
    free(io_str);
    node = isl_schedule_node_insert_mark(node, id);
    node = isl_schedule_node_parent(node);
    io_level++;
  }

  isl_mat_free(io_trans_mat);

  /* Store the I/O schedule */
  sched = isl_schedule_node_get_schedule(node);
  group->io_schedule = isl_schedule_dup(sched);
  group->io_trans = io_trans_ma;
  isl_schedule_free(sched);
  group->io_level = io_level;
  group->space_dim = space_dim;
  isl_schedule_node_free(node);

  return isl_stat_ok;
}

static isl_stat compute_io_group_buffer(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_gen *gen)
{
  isl_schedule_node *node;
  int io_level = group->io_level;
  int i;

  node = isl_schedule_get_root(group->io_schedule);

  /* Compute the group tiling at each I/O level */
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  i = 1;
  assert(group->io_buffers == NULL);
  assert(group->n_io_buffer == 0);
  group->io_buffers = NULL;
  group->n_io_buffer = 0;
  while (i <= io_level) {
    node = isl_schedule_node_parent(node);
    if (isl_schedule_node_is_io_mark(node, i)) {
      /* In the automatic mode, PolySA only computes the tiling at L1
       * for drain group and I/O group with interior I/O, and at L2 for I/O 
       * group with exterior I/O.
       */
      (group->n_io_buffer)++;
      group->io_buffers = (struct polysa_io_buffer **)realloc(group->io_buffers, sizeof(struct polysa_io_buffer *) * group->n_io_buffer);
      group->io_buffers[group->n_io_buffer - 1] = (struct polysa_io_buffer *)malloc(
          sizeof(struct polysa_io_buffer));
      group->io_buffers[group->n_io_buffer - 1]->level = i;
      if (group->group_type == POLYSA_DRAIN_GROUP) {
        if (i == 1) {
          /* Compute the group tiling at this level */
          compute_group_bounds_drain_at_node(kernel, group, node, group->io_buffers[group->n_io_buffer - 1]); 
          polysa_array_ref_group_compute_tiling(group->io_buffers[group->n_io_buffer - 1]->tile, group);
        } else {
          group->io_buffers[group->n_io_buffer - 1]->tile = NULL;
        }
      } else if (group->group_type == POLYSA_IO_GROUP) {
        if ((group->io_type == POLYSA_EXT_IO && i == 2) ||
           (group->io_type == POLYSA_INT_IO && i == 1)) {
          /* Compute the group tiling at this level */
          compute_group_bounds_io_at_node(kernel, group, node, group->io_buffers[group->n_io_buffer - 1]); 
          polysa_array_ref_group_compute_tiling(group->io_buffers[group->n_io_buffer - 1]->tile, group);
        } else {
          group->io_buffers[group->n_io_buffer - 1]->tile = NULL;
        }
      } else {
        group->io_buffers[group->n_io_buffer - 1]->tile = NULL;
      }
      i++;
    }
  }

  isl_schedule_node_free(node);

  return isl_stat_ok;
}

struct update_group_simd_data {
  struct polysa_array_ref_group *group;
  struct polysa_kernel *kernel;
};

/* Examine if there is any array references in the "group" under the SIMD loop.
 * If so, exmaine if the array reference has a stride of 1 under teh SIMD loop.
 * If so, update the SIMD lane of the "group".
 */
static isl_bool update_group_simd(__isl_keep isl_schedule_node *node, void *user)
{
  struct update_group_simd_data *data = user;
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  // debug

  if (isl_schedule_node_get_type(node) == isl_schedule_node_mark) {
    isl_id *id;
    isl_union_set *domain;
    struct polysa_array_ref_group *group = data->group;

    id = isl_schedule_node_mark_get_id(node);
    if (strcmp(isl_id_get_name(id), "simd")) {
      isl_id_free(id);
      return isl_bool_true;
    }
    
    isl_id_free(id);
    node = isl_schedule_node_child(node, 0);
    domain = isl_schedule_node_get_domain(node);
    for (int i = 0; i < group->n_ref; i++) {
      struct polysa_stmt_access *ref = group->refs[i];
      for (int j = 0; j < ref->n_io_info; j++) {
        struct polysa_io_info *info = ref->io_info[j];
        if (info->io_type == group->io_type && !isl_vec_cmp(info->dir, group->dir)) {
          /* Test if either the source or dest of the dependence associated with
           * the array reference is intersected with the current loop domain */
          struct polysa_dep *dep = info->dep;
          isl_basic_map *bmap;
          isl_map *map;
          isl_set *src, *dest;
          isl_union_set *uset;
//          // debug
//          p = isl_printer_print_basic_map(p, dep->isl_dep);
//          printf("\n");
//          // debug
          bmap = isl_basic_map_copy(dep->isl_dep);
          map = isl_map_from_basic_map(bmap);
          map = isl_map_factor_domain(map);
          src = isl_map_domain(isl_map_copy(map));
          dest = isl_map_range(map);
          uset = isl_union_set_union(isl_union_set_from_set(src), 
                    isl_union_set_from_set(dest));
          uset = isl_union_set_intersect(uset, isl_union_set_copy(domain));
          if (!isl_union_set_is_empty(uset)) {
            if (ref->simd_stride == 1)
              group->n_lane = data->kernel->simd_w;
          }
          isl_union_set_free(uset);
        }
      }
    }
    isl_union_set_free(domain);
  }

  return isl_bool_true;
}

/* Select the data pack factor for I/O buffers. The data pack factor
 * should be sub-multiples of the last dimension of the local array.
 * Meanwhile, it should also be sub-multiples of the data pack factors 
 * selected for the upper-level I/O buffers.
 * 
 * If SIMD vectorization is enabled, and the data stored in the I/O buffer is 
 * to be vectorized, the data pack factor should also be multiples of the SIMD factor.
 */
static isl_stat compute_io_group_data_pack(struct polysa_kernel *kernel,
  struct polysa_array_ref_group *group, struct polysa_gen *gen)
{
  isl_schedule_node *node; 
  struct update_group_simd_data data;
  int ele_size = group->array->size; // bytes
  /* Given the DRAM port width as 64 Bytes, compute the maximal data pack factor. */
  int max_n_lane = 64 / ele_size; 

  /* Examine if any of the array reference in the group is in used by SIMD loop.
   * The default SIMD lane for the group is 1. 
   * If any of the array references in the group is under the SIMD loop, and 
   * if the stride of reference under the loop is one. The SIMD lane of the 
   * group is then updated to the SIMD lane of the loop.
   */
  group->n_lane = 1;
  node = isl_schedule_get_root(kernel->schedule);
  data.group = group;
  data.kernel = kernel;
  isl_schedule_node_foreach_descendant_top_down(node, &update_group_simd, &data);
  isl_schedule_node_free(node);
  if (max_n_lane % group->n_lane != 0) {
    printf("[PolySA] The data is not aligned to the DRAM port with SIMD vectorization. Abort!\n");
    printf("[PolySA] Please try to use SIMD factor as sub-multiples of %d.\n", max_n_lane);
    exit(1);
  }

  if (!gen->options->data_pack) {
    for (int i = 0; i < group->io_level; i++) {
      struct polysa_io_buffer *buf = group->io_buffers[i];
      buf->n_lane = group->n_lane;
    }
    return isl_stat_ok;
  }

  int cur_n_lane = group->n_lane;
  int status = false;
  for (int i = 0; i < group->io_level; i++) {
    struct polysa_io_buffer *buf = group->io_buffers[i];
    if (buf->tile) {
      int n_lane = cur_n_lane;
      isl_val *size = isl_val_copy(buf->tile->bound[group->array->n_index - 1].size);
      while (n_lane <= max_n_lane) {
        /* The lane should be multiples of SIMD lane */
        if (n_lane % group->n_lane == 0) {
          isl_val *val = isl_val_int_from_si(gen->ctx, n_lane);      
          /* The lane should be sub-multiples of the last dim of the array */
          if (isl_val_is_divisible_by(size, val)) {
            cur_n_lane = n_lane;
            status = true;          
          }
          isl_val_free(val);
        }
        n_lane = n_lane * 2;
      }
      if (status) {
        buf->n_lane = cur_n_lane;
      } else {
        printf("[PolySA] Cannot find data pack factors as sub-multiples of the last dim of the local array. Abort!\n");
        printf("[PolySA] Please try to use different tiling factors.\n");
      }
      isl_val_free(size);
    } else {
      buf->n_lane = cur_n_lane;
    }
  }

  return isl_stat_ok;
}

/* "node" points to the space band node.
 * "space_dim" denotes the band width.
 * Return "level" denoting the number of I/O levels being clustered.
 * This function clusters I/O level by level and computes the array tiling
 * at each level.
 */
static isl_stat polysa_io_construct(
  struct polysa_kernel *kernel, struct polysa_array_ref_group *group, 
  struct polysa_gen *gen) 
{
  compute_io_group_schedule(kernel, group, gen);
  compute_io_group_buffer(kernel, group, gen);
  compute_io_group_data_pack(kernel, group, gen);

  return isl_stat_ok;
}

/* Group array references together if they share the same data transfer chain.
 * Return -1 on error.
 *
 * Two array references are grouped together if they share the same data 
 * transfer direction "dir" and I/O type "io_type".
 * Besides, they should all under the SIMD loop or not.
 *
 * For exterior I/O pair, calculate the group tiling at the io_L2 level.
 * For interior I/O pair, calculate the group tiling at the io_L1 level.
 */
static int group_array_references_io(struct polysa_kernel *kernel,
	struct polysa_local_array_info *local, struct polysa_group_data *data)
{
	int i, j;
	int n;
	isl_ctx *ctx = isl_union_map_get_ctx(data->pe_sched);
	struct polysa_array_ref_group **groups;

  /* Count the total number of groups. */
  n = 0;
  for (i = 0; i < local->array->n_ref; i++) {
    struct polysa_stmt_access *ref = local->array->refs[i];
    n += ref->n_io_info;
  }

  groups = (struct polysa_array_ref_group **)calloc(n, sizeof(struct polysa_array_ref_group *));
  if (!groups)
    return -1;

  /* Populate the groups */
	n = populate_array_references_io(local, groups, data);

  /* Group references that share the same data transfer direction and I/O type. */
  n = group_share_io(kernel, n, groups, data);

  /* Construct the I/O and compute the I/O buffers */
  for (i = 0; i < n; ++i) {
    polysa_io_construct(kernel, groups[i], data->gen);
  }

	for (i = 0; i < n; ++i) {
		if (compute_group_bounds_io(kernel, groups[i], data) < 0) { 
		  for (j = 0; j < n; ++j) {
        polysa_array_ref_group_free(groups[j]);
      }
      free(groups);
      return -1;
    }
  }

	set_array_groups_io(local, n, groups);

	return 0;
}

struct extract_access_waw_domain_data {
  struct polysa_stmt_access *ref;
  isl_set *drain_domain;
};

/* Check if the access is associated with the waw,
 * if so, calculate the write-out (drain) domain as:
 * acc domain - waw src_domain
 */
static void extract_access_waw_domain(__isl_keep isl_basic_map *dep, void *user)
{
  isl_space *space;
  isl_space *src_space;
  isl_id *src_id;
  isl_set *src_domain;
  struct extract_access_waw_domain_data *data = (struct extract_access_waw_domain_data *)(user);
  isl_basic_map *bmap;
  isl_map *map;

  space = isl_basic_map_get_space(dep); 
  src_space = isl_space_unwrap(isl_space_domain(space));
  src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
  isl_space_free(src_space);

  if (src_id != data->ref->ref_id) {
    isl_id_free(src_id);
    return;
  }
  isl_id_free(src_id);

  bmap = isl_basic_map_copy(dep);
  map = isl_map_from_basic_map(bmap);
  map = isl_map_factor_domain(map);
  src_domain = isl_map_domain(map);

  data->drain_domain = isl_set_subtract(data->drain_domain, src_domain);

  return;
}

isl_bool extract_access_waw_domain_wrap(__isl_keep isl_map *map, void *user)
{
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  for (int i = 0; i < isl_map_n_basic_map(map); i++) {
    isl_basic_map *dep = isl_basic_map_list_get_basic_map(bmap_list, i);
    extract_access_waw_domain(dep, user);
    isl_basic_map_free(dep);
  }
  isl_basic_map_list_free(bmap_list);
  return isl_bool_true;
}

/* Group array references together if they are associated with WAW dep and need 
 * to be drained out.
 * Return -1 on error.
 *
 * Calculate the group tiling at the PE level.
 */
static int group_array_references_drain(struct polysa_kernel *kernel,
  struct polysa_local_array_info *local, struct polysa_group_data *data)
{
  int i, j;
  int n;
  isl_ctx *ctx = isl_union_map_get_ctx(data->pe_sched);
  struct polysa_array_ref_group **groups = NULL;
  isl_union_map *dep_waw = kernel->scop->tagged_dep_waw;

  /* Populate the groups */
  n = 0;
  for (int i = 0; i < local->array->n_ref; ++i) {
    struct polysa_stmt_access *access = local->array->refs[i];
    if (access->read)
      continue;
    isl_set *domain = isl_map_domain(isl_map_copy(access->access)); 
    isl_set *access_domain = isl_union_set_extract_set( 
        kernel->expanded_domain, 
        isl_set_get_space(domain));
    isl_set_free(domain);
    struct extract_access_waw_domain_data drain_data = {access, access_domain};
    isl_union_map_every_map(dep_waw, &extract_access_waw_domain_wrap, &drain_data);
    if (!isl_set_is_empty(drain_data.drain_domain)) {
      isl_map *map;
      isl_union_map *umap;
      
      map = isl_map_copy(access->access);
      umap = isl_union_map_from_map(map);
      umap = isl_union_map_apply_domain(umap,
          isl_union_map_copy(data->pe_sched)); 

      map = isl_map_from_union_map(umap);
      map = isl_map_detect_equalities(map);  

      /* Add this access relation to the group */
      struct polysa_array_ref_group *group = isl_calloc_type(ctx, struct polysa_array_ref_group);
      if (!group) {
        isl_map_free(map);
        isl_set_free(drain_data.drain_domain);
        return -1;
      }

      group->local_array = local;
      group->array = local->array;
      group->access = map;
      group->write = access->write;
      group->exact_write = access->exact_write;
      group->slice = access->n_index < local->array->n_index;
      group->refs = &local->array->refs[i];
      group->n_ref = 1;
      group->io_type = POLYSA_INT_IO;
      group->dir = isl_vec_zero(ctx, kernel->n_sa_dim);
      group->dir = isl_vec_set_element_si(group->dir, 0, 1);
      group->group_type = POLYSA_DRAIN_GROUP;
      group->pe_io_dir = IO_OUT;
      group->array_io_dir = IO_OUT;
      group->io_pe_expr = NULL;
      group->io_L1_pe_expr = NULL;
      group->n_io_buffer = 0;
      group->io_buffers = NULL;
      group->copy_schedule = NULL;

      groups = (struct polysa_array_ref_group **)realloc(groups, (++n) * sizeof(struct polysa_array_ref_group *));
      groups[n - 1] = group;
    }
    isl_set_free(drain_data.drain_domain);
  }

  /* Join all referneces together */
  for (i = 1; i < n; ++i) {
    groups[0] = join_groups_and_free(groups[0], groups[i]);
  }
  if (n > 1)
    n = 1;

  /* Construct the I/O and compute the I/O buffers */
  for (i = 0; i < n; ++i) {
    polysa_io_construct(kernel, groups[i], data->gen);
  }

  /* Calculate the group tiling. */
  for (i = 0; i < n; ++i) {
    if (compute_group_bounds_drain(kernel, groups[i], data) < 0) {
      for (j = 0; j < n; j++) {
        polysa_array_ref_group_free(groups[j]);
      }
      free(groups);
      n = 0;
      return -1;
    }
  }

  /* Set the group. */
  if (n > 0) {
    groups[0]->nr = 0;
    local->drain_group = groups[0];
  } else {
    local->drain_group = NULL;
  }
  free(groups);
    
  return 0;
}

/* Print the name of the local copy of a given group of array references.
 */
__isl_give isl_printer *polysa_array_ref_group_print_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p)
{
	int global = 0;
  enum polysa_group_access_type type;

  type = polysa_array_ref_group_type(group);
  if (type == POLYSA_ACCESS_LOCAL)
    p = isl_printer_print_str(p, "local_");
  else
    global = 1;

  p = isl_printer_print_str(p, group->array->name);
  if (!global) {
    if (group->group_type == POLYSA_IO_GROUP && group->local_array->n_io_group > 1) {
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
    } else if (group->group_type == POLYSA_PE_GROUP && group->local_array->n_pe_group > 1) {
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
    }
  }

	return p;
}

/* Print the name of the local copy of a given group of array references.
 */
__isl_give isl_printer *polysa_array_ref_group_print_fifo_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p)
{
	int global = 0;
  enum polysa_group_access_type type;

  if (group->group_type == POLYSA_PE_GROUP)
    return p;

  p = isl_printer_print_str(p, "fifo_");
  p = isl_printer_print_str(p, group->array->name);
  if (group->local_array->n_io_group > 1) {
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_int(p, group->nr);
  }
  if (group->group_type == POLYSA_DRAIN_GROUP) {
    p = isl_printer_print_str(p, "_drain");
  }

	return p;
}

/* Given a description of an array tile "tile" and the "space"
 *
 *	{ D -> A }
 *
 * where D represents the first tile->depth schedule dimensions
 * and A represents the array, construct an isl_multi_aff
 *
 *	{ [D[i] -> A[a]] -> A'[a'] }
 *
 * with A' a scaled down copy of A according to the shifts and strides
 * in "tile".  In particular,
 *
 *	a' = (a + shift(i))/stride
 *
 * "insert_array" represents
 *
 *	{ [D -> A] -> D }
 *
 * and is used to insert A into the domain of functions that only
 * reference D.
 */
static __isl_give isl_multi_aff *strided_tile(
	struct polysa_array_tile *tile, __isl_keep isl_space *space,
	__isl_keep isl_multi_aff *insert_array)
{
	int i;
	isl_ctx *ctx;
	isl_multi_aff *shift;
	isl_multi_val *stride;
	isl_space *space2;
	isl_local_space *ls;
	isl_multi_aff *tiling;

	ctx = isl_space_get_ctx(space);
	space2 = isl_space_domain(isl_space_copy(space));
	ls = isl_local_space_from_space(space2);
	space2 = isl_space_range(isl_space_copy(space));
	stride = isl_multi_val_zero(space2);
	shift = isl_multi_aff_zero(isl_space_copy(space));

	for (i = 0; i < tile->n; ++i) {
		struct polysa_array_bound *bound = &tile->bound[i];
		isl_val *stride_i;
		isl_aff *shift_i;

		stride_i = isl_val_copy(bound->stride);
		shift_i = isl_aff_copy(bound->shift);

		stride = isl_multi_val_set_val(stride, i, stride_i);
		shift = isl_multi_aff_set_aff(shift, i, shift_i);
	}
	isl_local_space_free(ls);

	shift = isl_multi_aff_pullback_multi_aff(shift,
				    isl_multi_aff_copy(insert_array));

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_multi_aff(p, shift);
//  printf("\n");
//  p = isl_printer_print_multi_val(p, stride);
//  printf("\n");
//  p = isl_printer_print_space(p, space);
//  printf("\n");
//  // debug

	tiling = isl_multi_aff_range_map(isl_space_copy(space));
	tiling = isl_multi_aff_add(tiling, shift);
	tiling = isl_multi_aff_scale_down_multi_val(tiling, stride);

	return tiling;
}

static __isl_give isl_multi_aff *strided_tile_depth(
	struct polysa_array_tile *tile, __isl_keep isl_space *space,
	__isl_keep isl_multi_aff *insert_array, int depth)
{
	int i;
	isl_ctx *ctx;
	isl_multi_aff *shift;
	isl_multi_val *stride;
	isl_space *space2;
	isl_local_space *ls;
	isl_multi_aff *tiling;

	ctx = isl_space_get_ctx(space);
	space2 = isl_space_domain(isl_space_copy(space));
	ls = isl_local_space_from_space(space2);
	space2 = isl_space_range(isl_space_copy(space));
	stride = isl_multi_val_zero(space2);
	shift = isl_multi_aff_zero(isl_space_copy(space));

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  // debug

	for (i = 0; i < tile->n; ++i) {
		struct polysa_array_bound *bound = &tile->bound[i];
		isl_val *stride_i;
		isl_aff *shift_i;

		stride_i = isl_val_copy(bound->stride);
		shift_i = isl_aff_copy(bound->shift);

//    // debug
//    p = isl_printer_print_aff(p, shift_i);
//    printf("\n");
//    printf("%d\n", isl_aff_dim(shift_i, isl_dim_in));
//    // debug
    shift_i = isl_aff_insert_dims(shift_i, isl_dim_in, tile->depth, depth - tile->depth);
//    // debug
//    p = isl_printer_print_aff(p, shift_i);
//    printf("\n");
//    // debug

		stride = isl_multi_val_set_val(stride, i, stride_i);
		shift = isl_multi_aff_set_aff(shift, i, shift_i);
	}
	isl_local_space_free(ls);

	shift = isl_multi_aff_pullback_multi_aff(shift,
				    isl_multi_aff_copy(insert_array));

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_multi_aff(p, shift);
//  printf("\n");
//  p = isl_printer_print_multi_val(p, stride);
//  printf("\n");
//  p = isl_printer_print_space(p, space);
//  printf("\n");
//  // debug

	tiling = isl_multi_aff_range_map(isl_space_copy(space));
	tiling = isl_multi_aff_add(tiling, shift);
	tiling = isl_multi_aff_scale_down_multi_val(tiling, stride);

	return tiling;
}

/* Recompute the tiling by extending the scheduling domain to the "depth". */
__isl_give isl_multi_aff *polysa_array_ref_group_recompute_tiling(
  struct polysa_array_tile *tile,
  struct polysa_array_ref_group *group,
  int depth)
{
 	int i;
	isl_space *space;
	isl_multi_aff *tiling, *lb, *insert_array;
	isl_printer *p;
	char *local_name;

  if (tile == NULL)
    return NULL;

	space = isl_map_get_space(group->access);
	space = isl_space_from_range(isl_space_range(space));
  /* Build D[i] -> A[a] */
	space = isl_space_add_dims(space, isl_dim_in, depth);
  /* Build [D[i] -> A[a]] -> D[i] */
	insert_array = isl_multi_aff_domain_map(isl_space_copy(space));

	for (i = 0; i < tile->n; ++i)
		if (tile->bound[i].shift)
			break;

	if (i < tile->n)
		tiling = strided_tile_depth(tile, space, insert_array, depth); 
	else
		tiling = isl_multi_aff_range_map(isl_space_copy(space));  

	lb = isl_multi_aff_zero(space);
	for (i = 0; i < tile->n; ++i) {
		isl_aff *lb_i = isl_aff_copy(tile->bound[i].lb); 
    lb_i = isl_aff_insert_dims(lb_i, isl_dim_in, tile->depth, depth - tile->depth);
		lb = isl_multi_aff_set_aff(lb, i, lb_i);
	}
	lb = isl_multi_aff_pullback_multi_aff(lb, insert_array);

	tiling = isl_multi_aff_sub(tiling, lb);

	p = isl_printer_to_str(isl_multi_aff_get_ctx(tiling));
	p = polysa_array_ref_group_print_name(group, p);
	local_name = isl_printer_get_str(p);
	isl_printer_free(p);
	tiling = isl_multi_aff_set_tuple_name(tiling, isl_dim_out, local_name);
	free(local_name);

	return tiling;
}

/* Compute a tiling for the array reference group "group".
 *
 * The tiling is of the form
 *
 *	{ [D[i] -> A[a]] -> T[t] }
 *
 * where D represents the first tile->depth schedule dimensions,
 * A represents the global array and T represents the local memory 
 * tile.  The name of T is the name of the local array.
 *
 * If there is any stride in the accesses, then the mapping is
 *
 *	t = (a + shift(i))/stride - lb(i)
 *
 * otherwise, it is simply
 *
 *	t = a - lb(i)
 */
void polysa_array_ref_group_compute_tiling(
  struct polysa_array_tile *tile,
  struct polysa_array_ref_group *group)
{
	int i;
	isl_space *space;
	isl_multi_aff *tiling, *lb, *insert_array;
	isl_printer *p;
	char *local_name;

  if (tile == NULL && polysa_array_ref_group_tile(group) == NULL)
    return;

  if (tile == NULL)
    tile = polysa_array_ref_group_tile(group);

	space = isl_map_get_space(group->access);
	space = isl_space_from_range(isl_space_range(space));
  /* Build D[i] -> A[a] */
	space = isl_space_add_dims(space, isl_dim_in, tile->depth);
  /* Build [D[i] -> A[a]] -> D[i] */
	insert_array = isl_multi_aff_domain_map(isl_space_copy(space));

	for (i = 0; i < tile->n; ++i)
		if (tile->bound[i].shift)
			break;

	if (i < tile->n)
		tiling = strided_tile(tile, space, insert_array); 
	else
		tiling = isl_multi_aff_range_map(isl_space_copy(space));  

	lb = isl_multi_aff_zero(space);
	for (i = 0; i < tile->n; ++i) {
		isl_aff *lb_i = isl_aff_copy(tile->bound[i].lb);
		lb = isl_multi_aff_set_aff(lb, i, lb_i);
	}
	lb = isl_multi_aff_pullback_multi_aff(lb, insert_array);

	tiling = isl_multi_aff_sub(tiling, lb);

	p = isl_printer_to_str(isl_multi_aff_get_ctx(tiling));
	p = polysa_array_ref_group_print_name(group, p);
	local_name = isl_printer_get_str(p);
	isl_printer_free(p);
	tiling = isl_multi_aff_set_tuple_name(tiling, isl_dim_out, local_name);
	free(local_name);

	tile->tiling = tiling;
}

/* Group references of all arrays in "kernel".
 * Each array is associated with three types of groups:
 * PE group: Assign the local buffers inside PEs.
 * I/O group: Assign the I/O modules for transferring data between
 *   PEs and the external memory
 * Drain group: Assign the I/O modules for transferring out the results from
 *   PEs to the external memory.
 */
isl_stat sa_group_references(struct polysa_kernel *kernel, struct polysa_gen *gen)
{
  int r = 0;
  struct polysa_group_data data;
  isl_schedule_node *node;
  isl_union_pw_multi_aff *contraction;

  node = isl_schedule_get_root(kernel->schedule);
  node = polysa_tree_move_down_to_kernel(node);

  /* Set up polysa_group_data */
  data.scop = kernel->prog->scop;
  data.gen = gen;
  data.kernel_depth = isl_schedule_node_get_schedule_depth(node);
  data.host_sched = isl_schedule_node_get_prefix_schedule_relation(node);

//  node = polysa_tree_move_down_to_local(node, kernel->core);
//  data.local_depth = isl_schedule_node_get_schedule_depth(node);
//  data.local_sched = prefix_with_equalities(node);
  
  node = polysa_tree_move_down_to_pe(node, kernel->core);
  data.pe_depth = isl_schedule_node_get_schedule_depth(node);
  data.pe_sched = prefix_with_equalities(node);

  contraction = isl_union_pw_multi_aff_copy(kernel->contraction);
  data.host_sched = expand(data.host_sched, contraction);
//  data.local_sched = expand(data.local_sched, contraction);
  data.copy_sched = isl_union_map_copy(data.pe_sched); 
  data.pe_sched = expand(data.pe_sched, contraction);
  isl_union_pw_multi_aff_free(contraction);
  
  data.full_sched = isl_union_map_copy(data.pe_sched);
  data.full_sched = isl_union_map_flat_range_product(data.full_sched,
      isl_schedule_node_get_subtree_schedule_union_map(node));
  data.schedule = kernel->schedule;

  /* Group the array references for the PE */
  for (int i = 0; i < kernel->n_array; i++) {
    r = group_array_references_pe(kernel, &kernel->array[i], &data); 
    if (r < 0)
      break;
  }

  /* Group the array references for the I/O */
  for (int i = 0; i < kernel->n_array; i++) {
    r = group_array_references_io(kernel, &kernel->array[i], &data);
    if (r < 0)
      break;
  }

  /* Group the array references for the drain data */
  for (int i = 0; i < kernel->n_array; i++) {
    r = group_array_references_drain(kernel, &kernel->array[i], &data); 
    if (r < 0)
      break;
  }

  /* Copy the PE group results to the default groups, to
   * keep the default flow work. 
   */
  for (int i = 0; i < kernel->n_array; i++) {
    struct polysa_local_array_info *array = &kernel->array[i]; 
    array->n_group = array->n_pe_group;
    array->groups = array->pe_groups;
  }

  isl_union_map_free(data.host_sched);
//  isl_union_map_free(data.local_sched);
  isl_union_map_free(data.copy_sched);
  isl_union_map_free(data.full_sched);
  isl_union_map_free(data.pe_sched);
  isl_schedule_node_free(node);
  
  return isl_stat_ok;
}

/* For each reference, if 
 * - extract copy-in access (read == 1) 
 *   - read access
 *     - RAR: extract the union of the src and dest domain of dep
 *     - RAW: extract the dest domain of dep
 * - extract copy-out access (write == 1)
 *   - write access
 *     - RAW: extract the src domain of dep 
 */
__isl_give isl_union_map *polysa_io_group_access_relation(
  struct polysa_array_ref_group *group, int read, int write)
{
  isl_union_map *access;

  access = isl_union_map_empty(isl_map_get_space(group->access));
  for (int i = 0; i < group->n_ref; ++i) {
    struct polysa_stmt_access *ref_i = group->refs[i]; 

    if (!((read && group->refs[i]->read) ||
        (write && group->refs[i]->write)))
      continue;
   
    access = isl_union_map_union(access,
        polysa_io_group_ref_access_relation(group, ref_i, read, write));
  }

  /* Simplify the access relation. */
  access = isl_union_map_coalesce(access);

  return access;
}

__isl_give isl_union_map *polysa_drain_group_access_relation(
  struct polysa_array_ref_group *group, int read, int write,
  __isl_keep isl_union_set *domain)
{
  isl_union_map *access;

  access = isl_union_map_empty(isl_map_get_space(group->access)); 
  for (int i = 0; i < group->n_ref; ++i) {
    isl_map *map_i;
    struct polysa_stmt_access *ref_i = group->refs[i];
    isl_set *acc_domain;
    isl_space *space;
    isl_set *write_out;

    if (!((read && group->refs[i]->read) ||
          (write && group->refs[i]->write)))
      continue;
        
    acc_domain = isl_map_domain(isl_map_copy(ref_i->access)); 
    space = isl_set_get_space(acc_domain); 
    isl_set_free(acc_domain);
    acc_domain = isl_union_set_extract_set(domain, space); 
    for (int j = 0; j < ref_i->n_io_info; j++) {
      struct polysa_io_info *info_i = ref_i->io_info[j];
      if (info_i->dep->type == POLYSA_DEP_WAW) {
        isl_set *src_domain;

        isl_space *space = isl_basic_map_get_space(info_i->dep->isl_dep);
        isl_space *src_space = isl_space_unwrap(isl_space_domain(space)); 
        isl_id *src_id = isl_space_get_tuple_id(src_space, isl_dim_out); 
        isl_space_free(src_space);

        if (src_id != ref_i->ref_id) {
          isl_id_free(src_id);
          continue;        
        }
        isl_id_free(src_id);

        src_domain = isl_map_domain(isl_map_factor_domain(isl_map_from_basic_map(
                isl_basic_map_copy(info_i->dep->isl_dep)))); 
        acc_domain = isl_set_subtract(acc_domain, src_domain);
      }      
    }
    write_out = acc_domain;

    access = isl_union_map_union(access, 
        isl_union_map_from_map(isl_map_intersect_domain(isl_map_copy(ref_i->access), write_out)));
  }

  return access;
}

/* Return the access relation associated with the comm pair of the array reference
 * "ref" in the current I/O group "group".
 */
__isl_give isl_union_map *polysa_io_group_ref_access_relation(
  struct polysa_array_ref_group *group,
  struct polysa_stmt_access *ref,
  int read, int write)
{
  isl_union_map *access;
  isl_map *map;

  access = isl_union_map_empty(isl_map_get_space(ref->access));
  for (int i = 0; i < ref->n_io_info; i++) {
    struct polysa_io_info *info_i = ref->io_info[i];
    if (info_i->io_type == group->io_type &&
        !isl_vec_cmp(info_i->dir, group->dir))
    {
      isl_map *dep = isl_map_factor_domain(isl_map_from_basic_map(isl_basic_map_copy(info_i->dep->isl_dep)));
      isl_set *dep_src = isl_map_domain(isl_map_copy(dep));
      isl_set *dep_dest = isl_map_range(dep);
      if (info_i->dep->type == POLYSA_DEP_RAR) {
        isl_set *domain = isl_set_union(dep_src, dep_dest);
        domain = isl_set_coalesce(domain);
        access = isl_union_map_union(access,
            isl_union_map_from_map(isl_map_intersect_domain(
                isl_map_copy(ref->access), domain)));
      } else if (info_i->dep->type == POLYSA_DEP_RAW) {
        isl_set *domain;
        if (ref->read) {
          domain = dep_dest;
          isl_set_free(dep_src);
        } else {
          domain = dep_src;    
          isl_set_free(dep_dest);
        }
        access = isl_union_map_union(access,
            isl_union_map_from_map(isl_map_intersect_domain(
                isl_map_copy(ref->access), domain)));
      } else {
        isl_set_free(dep_src);
        isl_set_free(dep_dest);
      }
    }
  }

  return access;
}

__isl_give isl_union_map *polysa_drain_group_ref_access_relation(
  struct polysa_array_ref_group *group,
  struct polysa_stmt_access *ref,
  int read, int write, __isl_keep isl_union_set *domain)
{
  isl_union_map *access;
  isl_set *acc_domain;
  isl_space *space;

  access = isl_union_map_empty(isl_map_get_space(group->access));
  acc_domain = isl_map_domain(isl_map_copy(ref->access));
  space = isl_set_get_space(acc_domain);
  isl_set_free(acc_domain);
  acc_domain = isl_union_set_extract_set(domain, space);
  for (int i = 0; i < ref->n_io_info; i++) {
    struct polysa_io_info *info_i = ref->io_info[i];
    if (info_i->dep->type == POLYSA_DEP_WAW) {
      isl_set *src_domain;
      isl_space *space, *src_space;
      isl_id *src_id;

      space = isl_basic_map_get_space(info_i->dep->isl_dep);
      src_space = isl_space_unwrap(isl_space_domain(space));
      src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
      isl_space_free(src_space);
      if (src_id != ref->ref_id) {
        isl_id_free(src_id);
        continue;
      }
      isl_id_free(src_id);
      src_domain = isl_map_domain(isl_map_factor_domain(isl_map_from_basic_map(
              isl_basic_map_copy(info_i->dep->isl_dep))));
      acc_domain = isl_set_subtract(acc_domain, src_domain);
    }
  }
  access = isl_union_map_union(access,
    isl_union_map_from_map(isl_map_intersect_domain(isl_map_copy(ref->access), acc_domain)));

  return access;
}
