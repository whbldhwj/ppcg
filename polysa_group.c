#include "polysa_group.h"
#include "polysa_codegen.h"
#include "polysa_array_tile.h"

/* Internal data structure for polysa_group_references.
 */
struct polysa_group_data {
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
	// int thread_depth;
	// int n_thread;
	// isl_set *privatization;
  
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
static int populate_array_references(struct polysa_local_array_info *local,
	struct polysa_array_ref_group **groups, struct polysa_group_data *data)
{
	int i;
	int n;
	isl_ctx *ctx = isl_union_map_get_ctx(data->copy_sched);

	n = 0;
	for (i = 0; i < local->array->n_ref; ++i) {
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
		group->access = map;
		group->write = access->write;
		group->exact_write = access->exact_write;
		group->slice = access->n_index < local->array->n_index;
		group->refs = &local->array->refs[i];
		group->n_ref = 1;

		groups[n++] = group;
	}

	return n;
}

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
	//gpu_array_tile_free(group->shared_tile);
	//gpu_array_tile_free(group->private_tile);
	isl_map_free(group->access);
	if (group->n_ref > 1)
		free(group->refs);
	free(group);
	return NULL;
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
  /* Add dim_in that equates the parameters "ids". */
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

/* Map the domain of "access" to the outer data->local_depth
 * schedule dimensions.   
 */
static __isl_give isl_map *local_access(struct polysa_array_ref_group *group,
	__isl_keep isl_union_map *access, struct polysa_group_data *data)
{
	isl_union_map *local;

  local = isl_union_map_copy(access);
  local = isl_union_map_apply_domain(local,
      isl_union_map_copy(data->local_sched));
  return isl_map_from_union_map(local);

//	if (data->shared_depth == data->thread_depth)
//		return isl_map_copy(group->access);
//
//	shared = isl_union_map_copy(access);
//	shared = isl_union_map_apply_domain(shared,
//			isl_union_map_copy(data->shared_sched));
//	return isl_map_from_union_map(shared);
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
static isl_bool can_tile(__isl_keep isl_map *access,
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

	valid = isl_fixed_box_is_valid(box);
	if (valid >= 0 && valid) {
		offset = isl_fixed_box_get_offset(box);
		size = isl_fixed_box_get_size(box);
		for (i = 0; i < tile->n; ++i) {
			tile->bound[i].size = isl_multi_val_get_val(size, i);
			tile->bound[i].lb = isl_multi_aff_get_aff(offset, i);
		}
		isl_multi_aff_free(offset);
		isl_multi_val_free(size);
	}
	isl_fixed_box_free(box);

	return valid;
}

/* Compute the local memory tiles for the array reference group "group" 
 * of array "array". Return isl_stat_ok on success and isl_stat_error on error.
 *
 * If the array is a read-only scalar or if the user requested not to use
 * local memory, then we do not need to do anything.
 *
 * For FPGA, we will compute a local memory tile for every reference group.
 */
static isl_stat compute_group_bounds_core(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
	isl_ctx *ctx = isl_space_get_ctx(group->array->space);
	isl_union_map *access, *local;
	int n_index = group->array->n_index;
	int no_reuse, coalesced;
	isl_map *acc;
	// int force_private = group->local_array->force_private;
	// int use_shared = !force_private && kernel->options->use_shared_memory &&
	// 			data->n_thread > 0;
	// int use_private = force_private || kernel->options->use_private_memory;
  int use_local = kernel->options->use_local_memory;
	isl_stat r = isl_stat_ok;
	isl_bool ok;
	int requires_unroll;
	int unique_depth;

  if (!use_local)
    return isl_stat_ok;
  if (polysa_array_is_read_only_scalar(group->array))
    return isl_stat_ok;
  /* PPCG: If the array group involves any may writes (that are not must writes),
   * then we would have to make sure that we load the data into local memory first
   * in case the data is not written by the kernel (but still written back out to global
   * memory).
   * Since we don't have any such mechanism at the moment, we don't compute 
   * local tiles for groups involving may writes.
   */
  if (!group->exact_write)
    return isl_stat_ok;
  /* PPCG: If any reference in the reference group accesses more than one element,
   * then we would have to make sure that the layout in shared memory is the same as that
   * in global memory. Since we do not handle this yet (and it may not even be possible),
   * we refuse to map to local memory in such cases.
   */
  if (group->slice)
    return isl_stat_ok;

	access = polysa_array_ref_group_access_relation(group, 1, 1); 
	local = localize_access(data, isl_union_map_copy(access));
	no_reuse = isl_union_map_is_injective(local);
	//if (no_reuse < 0)
	//	r = isl_stat_error;
	//if (use_shared && no_reuse)
	//	coalesced = access_is_coalesced(data, local);
  isl_union_map_free(local);

	//if (r >= 0 && kernel->options->debug->verbose &&
	//    use_shared && no_reuse && coalesced)
	//	report_no_reuse_and_coalesced(kernel, access);

  /* Create local tile */
  if (use_local) {
    /* Create a tile. */
    group->local_tile = polysa_array_tile_create(ctx,
            group->array->n_index);
    acc = local_access(group, access, data);
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
    
  /* Calculation for private tile, commented out */
  /* PPCG: 
   * For computing a private memory tile, we also require that there is
   * some reuse.  Moreover, we require that the access is private
   * to the thread.  That is, we check that any given array element
   * is only accessed by a single thread.
   * We compute an access relation that maps the outer
   * data->thread_depth + data->n_thread schedule dimensions.
   * The latter data->n_thread will be mapped to thread identifiers.
   * We actually check that those iterators that will be wrapped
   * partition the array space.  This check is stricter than necessary
   * since several iterations may be mapped onto the same thread
   * and then they could be allowed to access the same memory elements,
   * but our check does not allow this situation.
   *
   * For private memory tiles, the number of schedule dimensions that
   * affect the offset is computed and stored in tile->depth, with
   * a lower bound of data->kernel_depth.  If this depth is smaller
   * than the minimal depth that still ensures that every element
   * is accessed by a single thread, then the depth is raised
   * to this minimal depth.
   * The fields of the tile are then adjusted to only refer to the tile->depth
   * outer schedule dimensions.
   *
   * We also check that the index expression only depends on parallel
   * loops.  That way, we can move those loops innermost and unroll them.
   * Again, we use a test that is stricter than necessary.
   * We actually check whether the index expression only depends
   * on the iterators that are wrapped over the threads.
   * These are necessarily parallel, but there may be more parallel loops.
   *
   * Combining the injectivity of the first test with the single-valuedness
   * of the second test, we simply test for bijectivity.
   *
   * If the use of the private tile requires unrolling, but some
   * of the other arrays are forcibly mapped to private memory,
   * then we do not allow the use of this private tile since
   * we cannot move the schedule dimensions that need to be unrolled down
   * without performing some kind of expansion on those arrays
   * that are forcibly mapped to private memory.
   *
   * If the array is marked force_private, then we bypass all checks
   * and assume we can (and should) use registers only.
   *
   * If it turns out we can (or have to) use registers, we compute
   * the private memory tile size using can_tile, after introducing a dependence
   * on the thread indices.
   */

//	access = isl_union_map_apply_domain(access,
//					isl_union_map_copy(data->thread_sched));
//
//	acc = isl_map_from_union_map(access);
//
//	if (!force_private && !access_is_bijective(data, acc)) {
//		isl_map_free(acc);
//		return isl_stat_ok;
//	}
//
//	unique_depth = compute_accessed_by_single_thread_depth(data, acc);
//
//	acc = isl_map_intersect_domain(acc, isl_set_copy(data->privatization));
//	acc = isl_map_project_out(acc, isl_dim_in, data->thread_depth,
//								data->n_thread);
//	requires_unroll = check_requires_unroll(data, acc, force_private);
//	if (unique_depth < 0 || requires_unroll < 0 ||
//	    (requires_unroll && kernel->any_force_private)) {
//		isl_map_free(acc);
//		return requires_unroll < 0 ? isl_stat_error : isl_stat_ok;
//	}
//
//	group->private_tile = gpu_array_tile_create(ctx, n_index);
//	group->private_tile->requires_unroll = requires_unroll;
//	ok = can_tile(acc, group->private_tile);
//	if (ok >= 0 && !ok)
//		group->private_tile = gpu_array_tile_free(group->private_tile);
//	isl_map_free(acc);
//	if (ok < 0)
//		return isl_stat_error;
//
//	if (group->private_tile) {
//		struct gpu_array_tile *tile = group->private_tile;
//		int tile_depth = compute_tile_depth(data, tile);
//		if (tile_depth < unique_depth)
//			tile_depth = unique_depth;
//		if (tile_adjust_depth(tile, tile_depth) < 0)
//			return isl_stat_error;
//	}
//
//	if (force_private && !group->private_tile)
//		isl_die(ctx, isl_error_internal,
//			"unable to map array reference group to registers",
//			return isl_stat_error);

	return isl_stat_ok;
}

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

/* Determine the number of schedule dimensions that affect the offset of the
 * local tile and store it in group->min_depth, with a lower bound of data->kernel_depth.
 * If there is no tile defined on the array reference group,
 * then set group->min_depth to data->local_depth.
 */
static int set_depth(struct polysa_group_data *data,
	struct polysa_array_ref_group *group)
{
	group->min_depth = data->local_depth;

	if (group->local_tile) {
		if (tile_set_depth(data, group->local_tile) < 0) 
			return -1;
		if (group->local_tile->depth < group->min_depth)
			group->min_depth = group->local_tile->depth;
	}

	return 0;
}

/* Compute the local memory tiles for the array
 * reference group "group" of array "array" and set the tile depth.
 * Return 0 on success and -1 on error.
 */
static int compute_group_bounds(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, struct polysa_group_data *data)
{
	if (!group)
		return -1;
	if (compute_group_bounds_core(kernel, group, data) < 0) 
		return -1;
	if (set_depth(data, group) < 0) 
		return -1;

	return 0;
}

/* If two groups have overlapping access relations (as determined by
 * the "overlap" function) and if one of them involves a write,
 * then merge the two groups into one.
 * If "compute_bounds" is set, then call compute_group_bounds
 * on the merged groups.
 * If any group is merged into the current group, then its access
 * relation may have changed or it may have been turned into a write.
 * The combined group might therefore overlap with groups that
 * the original group did not overlap with. The groups therefore
 * need to be checked again.
 *
 * Return the updated number of groups.
 * Return -1 on error.
 */
static int group_writes(struct polysa_kernel *kernel,
	int n, struct polysa_array_ref_group **groups,
	int (*overlap)(struct polysa_array_ref_group *group1,
		struct polysa_array_ref_group *group2), int compute_bounds,
	struct polysa_group_data *data)
{
	int i, j;
	int any_merge;

	for (i = 0; i < n; i += !any_merge) {
		any_merge = 0;
		for (j = n - 1; j > i; --j) {
			if (!groups[i]->write && !groups[j]->write)
				continue;

			if (!overlap(groups[i], groups[j]))
				continue;

			any_merge = 1;
			groups[i] = join_groups_and_free(groups[i], groups[j]); 
			if (j != n - 1)
				groups[j] = groups[n - 1];
			groups[n - 1] = NULL;
			n--;

			if (!groups[i])
				return -1;
			if (compute_bounds &&
			    compute_group_bounds(kernel, groups[i], data) < 0) 
				return -1;
		}
	}

	return n;
}

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

/* If two groups have overlapping access relations (within the innermost
 * loop) and if one of them involves a write, then merge the two groups
 * into one.
 *
 * Return the updated number of groups.
 */
static int group_overlapping_writes(struct polysa_kernel *kernel,
	int n, struct polysa_array_ref_group **groups,
	struct polysa_group_data *data)
{
	return group_writes(kernel, n, groups, &accesses_overlap, 0, data);
}

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

/* If two groups have overlapping access relations (within the outer
 * depth loops) and if one of them involves a write,
 * then merge the two groups into one.
 *
 * Return the updated number of groups.
 */
static int group_depth_overlapping_writes(struct polysa_kernel *kernel,
	int n, struct polysa_array_ref_group **groups, struct polysa_group_data *data)
{
	return group_writes(kernel, n, groups, &depth_accesses_overlap, 1,
				data);
}

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

/* Given an initial grouping of array references and local memory tiles
 * for each group that allows for a local memory tile, merge two groups
 * if both have a local memory tile, the merged group also has
 * a local memory tile and the size of the tile for the merge group
 * is smaller than the sum of the tile sizes of the individual groups.
 * If any group is merged into the current group, then it may become
 * profitable to combine it with groups that were considered before
 * the merge.  The groups are therefore checked again after a merge.
 *
 * If merging two groups decreases the depth of the tile of
 * one or both of the two groups, then we need to check for overlapping
 * writes again.
 *
 * Return the number of groups after merging.
 * Return -1 on error.
 */
static int group_common_local_memory_tile(struct polysa_kernel *kernel,
	struct polysa_array_info *array, int n,
	struct polysa_array_ref_group **groups, struct polysa_group_data *data)
{
	int i, j;
	int recompute_overlap = 0;
	int any_merge;

	for (i = 0; i < n; i += !any_merge) {
		any_merge = 0;
		if (!groups[i]->shared_tile)
			continue;
		for (j = n - 1; j > i; --j) {
			struct polysa_array_ref_group *group;

			if (!groups[j]->local_tile)
				continue;

			if (!depth_accesses_overlap(groups[i], groups[j]))
				continue;

			group = join_groups(groups[i], groups[j]);
			if (compute_group_bounds(kernel, group, data) < 0) {
				polysa_array_ref_group_free(group);
				return -1;
			}
			if (!group->local_tile ||
			    !smaller_tile(group->local_tile,
					groups[i]->local_tile,
					groups[j]->local_tile)) {
				polysa_array_ref_group_free(group);
				continue;
			}

			any_merge = 1;
			if (group->min_depth < groups[i]->min_depth ||
			    group->min_depth < groups[j]->min_depth)
				recompute_overlap = 1;
			polysa_array_ref_group_free(groups[i]);
			polysa_array_ref_group_free(groups[j]);
			groups[i] = group;
			if (j != n - 1)
				groups[j] = groups[n - 1];
			n--;
		}
	}

	if (recompute_overlap)
		n = group_depth_overlapping_writes(kernel, n, groups, data);
	return n;
}

/* Set array->n_group and array->groups to n and groups.
 *
 * Additionally, set the "nr" field of each group.
 */
static void set_array_groups(struct polysa_local_array_info *array,
	int n, struct polysa_array_ref_group **groups)
{
	int i;

	array->n_group = n;
	array->groups = groups;

	for (i = 0; i < n; ++i)
		groups[i]->nr = i;
}

/* Group array referneces that should be considered together when
 * deciding whether to access them from local or global memory.
 * Return -1 on error.
 *
 * In particular, if two array references overlap and if one of them 
 * is a write, then the two references are grouped together.
 * We first perform an initial grouping based only on the access relation.
 * After computing local memory tiles, we check for overlapping writes 
 * again, but this time taking into accout the depth of the effective tile.
 *
 * Furthermore, if two groups admit a local memory tile and if the combination
 * of the two also admits a local memory tile, we merge the two groups.
 */
static int group_array_references(struct polysa_kernel *kernel,
	struct polysa_local_array_info *local, struct polysa_group_data *data)
{
	int i;
	int n;
	isl_ctx *ctx = isl_union_map_get_ctx(data->local_sched);
	struct polysa_array_ref_group **groups;

	groups = isl_calloc_array(ctx, struct polysa_array_ref_group *,
					local->array->n_ref);
	if (!groups)
		return -1;

	n = populate_array_references(local, groups, data);

  /* PPCG: if the array contains structures, compute a single
   * reference group without trying to find any tiles except when 
   * those arrays of structures are required to be mapped to private memory.
   */
//	if (local->array->has_compound_element && !local->force_private) {
//		n = join_all_groups(n, groups);
//		set_array_groups(local, n, groups);
//		return 0;
//	}

	n = group_overlapping_writes(kernel, n, groups, data);

	for (i = 0; i < n; ++i)
		if (compute_group_bounds(kernel, groups[i], data) < 0) 
			n = -1;

	n = group_depth_overlapping_writes(kernel, n, groups, data);

	n = group_common_local_memory_tile(kernel, local->array, 
					    n, groups, data);

	set_array_groups(local, n, groups);

	if (n >= 0)
		return 0;

	for (i = 0; i < local->array->n_ref; ++i)
		polysa_array_ref_group_free(groups[i]);
	return -1;
}

/* Group references of all arrays in "kernel".
 * "node" points to the kernel mark.
 * The mapping to local memory is computed at the "shared" mark.
 *
 * We first extract all required schedule information into
 * a polysa_group_data structure and then consider each array in turn.
 */
int polysa_group_references(struct polysa_kernel *kernel,
  __isl_keep isl_schedule_node *node)
{
  int i;
  int r = 0;
  isl_union_pw_multi_aff *contraction;
  struct polysa_group_data data;

  /* Set up polysa_group_data */
  data.scop = kernel->prog->scop;
  data.kernel_depth = isl_schedule_node_get_schedule_depth(node);
  data.host_sched = isl_schedule_node_get_prefix_schedule_relation(node);

  node = isl_schedule_node_copy(node);
  node = polysa_tree_move_down_to_local(node, kernel->core); 
  data.local_depth = isl_schedule_node_get_schedule_depth(node);
  data.local_sched = prefix_with_equalities(node);

  contraction = isl_union_pw_multi_aff_copy(kernel->contraction);
  data.host_sched = expand(data.host_sched, contraction);
  data.local_sched = expand(data.local_sched, contraction); 
  data.copy_sched = isl_union_map_copy(data.local_sched);

  isl_union_pw_multi_aff_free(contraction);
  data.full_sched = isl_union_map_copy(data.local_sched);
  data.full_sched = isl_union_map_flat_range_product(data.full_sched,
      isl_schedule_node_get_subtree_schedule_union_map(node));
  isl_schedule_node_free(node);

  for (i = 0; i < kernel->n_array; i++) {
    r = group_array_references(kernel, &kernel->array[i], &data); 
    if (r < 0)
      break;
  }

  isl_union_map_free(data.host_sched);
  isl_union_map_free(data.local_sched);
  isl_union_map_free(data.copy_sched);
  isl_union_map_free(data.full_sched);

  return r;
}

/* Print the name of the local copy of a given group of array references.
 */
__isl_give isl_printer *polysa_array_ref_group_print_name(
	struct polysa_array_ref_group *group, __isl_take isl_printer *p)
{
	int global = 0;
	enum polysa_group_access_type type;

	type = polysa_array_ref_group_type(group);
//	if (type == ppcg_access_private)
//		p = isl_printer_print_str(p, "private_");
//	else if (type == ppcg_access_shared)
//		p = isl_printer_print_str(p, "shared_");
//	else
//		global = 1;
  if (type == POLYSA_ACCESS_LOCAL)
    p = isl_printer_print_str(p, "local_");
  else
    global = 1;
  p = isl_printer_print_str(p, group->array->name);
	if (!global && group->local_array->n_group > 1) {
		p = isl_printer_print_str(p, "_");
		p = isl_printer_print_int(p, group->nr);
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

	tiling = isl_multi_aff_range_map(isl_space_copy(space));
	tiling = isl_multi_aff_add(tiling, shift);
	tiling = isl_multi_aff_scale_down_multi_val(tiling, stride);

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
void polysa_array_ref_group_compute_tiling(struct polysa_array_ref_group *group)
{
	int i;
	struct polysa_array_tile *tile;
	isl_space *space;
	isl_multi_aff *tiling, *lb, *insert_array;
	isl_printer *p;
	char *local_name;

	tile = polysa_array_ref_group_tile(group);
	if (!tile)
		return;

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
