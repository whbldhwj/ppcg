#include "polysa_codegen.h"
#include "polysa_array_tile.h"
#include "polysa_group.h"

/* Internal data structure for at_domain.
 * "prog" represents the entire scop.
 * "kernel" points to the kernel to which the current schedule node
 * belongs. It is set by before_mark and reset by after_mark.
 * It may be NULL if we are outside any kernel.
 */
struct polysa_at_domain_data {
  struct polysa_prog *prog;
  struct polysa_kernel *kernel;
  struct polysa_hw_module *module;
  struct polysa_hw_top_module *top;
  int filter_buffer;

  /* Under a "pipeline" mark */
  int under_pipeline;
  /* Under a "unroll" mark */
  int under_unroll;
  /* Inside a "pipeline" for loop */
  int in_pipeline_for;
  /* Inside a "unroll" for loop */
  int in_unroll_for;
};

/* Internal data structure for the index and AST expression transformation
 * callbacks for pet_stmt_build_ast_exprs.
 *
 * "kernel" is the kernel for which are computing AST expressions and
 * may be NULL if we are not inside a kernel.
 * "accesses" is the list of polysa_stmt_access in the statement.
 * "iterator_map" expresses the statement iterators in terms of
 * the AST loop iterators.
 * "sched2copy" expresses the outer copy_schedule_dim dimensions of
 * the kernel schedule in terms of the AST loop iterators and
 * may be NULL if we are not inside a kernel.
 *
 * The following fields are set in transform_index and used in transform_expr.
 * "array" is the array that is being accessed.
 * "global" is set if the global array is accessed (rather than
 * shared/private memory).
 * "local_array" refers to information on the array specialized
 * to the current kernel.
 */
struct polysa_transform_data {
	struct polysa_kernel *kernel;
	struct polysa_stmt_access *accesses;
	isl_pw_multi_aff *iterator_map;
	isl_pw_multi_aff *sched2copy;

	struct polysa_array_info *array;
	int global;
  int reg;
	struct polysa_local_array_info *local_array;
  struct polysa_array_ref_group *group;
};

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

/* Given a mapping "iterator_map" from the AST schedule to a domain,
 * return the corresponding mapping from the AST schedule
 * to the outer kernel->copy_schedule_dim dimensions of
 * the schedule computed by PolySA for this kernel.
 *
 * Note that kernel->copy_schedule_dim is at least as large as
 * the largest depth of any array reference group associated to the kernel.
 * This is needed as the returned schedule is used to extract a mapping
 * to the outer tile->depth dimensions in transform_index.
 */
static __isl_give isl_pw_multi_aff *compute_sched_to_copy(
	struct polysa_kernel *kernel, __isl_take isl_pw_multi_aff *iterator_map)
{
	isl_union_pw_multi_aff *upma;
	isl_pw_multi_aff *pma;
	isl_space *space;

	space = isl_space_range(isl_pw_multi_aff_get_space(iterator_map));
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out,
					kernel->copy_schedule_dim);

	upma = isl_union_pw_multi_aff_copy(kernel->copy_schedule);
	pma = isl_union_pw_multi_aff_extract_pw_multi_aff(upma, space);
	isl_union_pw_multi_aff_free(upma);

	return isl_pw_multi_aff_pullback_pw_multi_aff(pma, iterator_map);
}

static __isl_give isl_pw_multi_aff *compute_sched_to_copy_group(
  __isl_take isl_pw_multi_aff *iterator_map, struct polysa_array_ref_group *group)
{
  isl_union_pw_multi_aff *upma;
  isl_pw_multi_aff *pma;
  isl_space *space;

  space = isl_space_range(isl_pw_multi_aff_get_space(iterator_map));
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out,
            group->copy_schedule_dim);
  
  upma = isl_union_pw_multi_aff_copy(group->copy_schedule);
  pma = isl_union_pw_multi_aff_extract_pw_multi_aff(upma, space);
  isl_union_pw_multi_aff_free(upma);

  return isl_pw_multi_aff_pullback_pw_multi_aff(pma, iterator_map);
}

/* Return the polysa_stmt_access in the list "accesses" that corresponds
 * to "ref_id".
 */
static struct polysa_stmt_access *find_access(struct polysa_stmt_access *accesses,
	__isl_keep isl_id *ref_id)
{
	struct polysa_stmt_access *access;

	for (access = accesses; access; access = access->next)
		if (access->ref_id == ref_id)
			return access;

	return NULL;
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

/* Return the index of the array called "name" in the list of arrays.
 */
static int find_array_index(struct polysa_kernel *kernel, const char *name)
{
	int i;

	for (i = 0; i < kernel->n_array; ++i)
		if (!strcmp(name, kernel->array[i].array->name))
			return i;

	return -1;
}

/* Return a pointer to the polysa_array_ref_group in "local"
 * that contains the reference "access".
 * Return NULL if no such group can be found.
 */
static struct polysa_array_ref_group *find_ref_group(
	struct polysa_local_array_info *local, struct polysa_stmt_access *access)
{
	int i, j;

	for (i = 0; i < local->n_group; ++i) {
		struct polysa_array_ref_group *group = local->groups[i];

		for (j = 0; j < group->n_ref; ++j)
			if (group->refs[j] == access)
				return group;
	}

	return NULL;
}

/* Return a pointer to the polysa_array_ref_group in "local"
 * that contains the reference "access".
 * Return NULL if no such group can be found.
 */
static struct polysa_array_ref_group *find_ref_group_module(
	struct polysa_local_array_info *local, struct polysa_stmt_access *access)
{
	int i, j;

	for (i = 0; i < local->n_pe_group; ++i) {
		struct polysa_array_ref_group *group = local->pe_groups[i];

		for (j = 0; j < group->n_ref; ++j)
			if (group->refs[j] == access)
				return group;
	}

	return NULL;
}

/* Should this array reference group be mapped to private, shared or global
 * memory?
 * If we have computed both a private and a shared tile, then
 * the tile with the smallest depth is used.  If both have the same depth,
 * then the private tile is used.
 */
enum polysa_group_access_type polysa_cpu_array_ref_group_type(
	struct polysa_array_ref_group *group)
{
  if (group->local_tile)
    return POLYSA_ACCESS_LOCAL;
  return POLYSA_ACCESS_GLOBAL;
}

/* Should this array reference group be mapped to private, shared or global
 * memory?
 * If we have computed both a private and a shared tile, then
 * the tile with the smallest depth is used.  If both have the same depth,
 * then the private tile is used.
 */
enum polysa_group_access_type polysa_array_ref_group_type(
	struct polysa_array_ref_group *group)
{
  if (polysa_array_is_read_only_scalar(group->array))
    return POLYSA_ACCESS_GLOBAL;
  else
    return POLYSA_ACCESS_LOCAL;
}

/* Return the effective gpu_array_tile associated to "group" or
 * NULL if there is no such polysa_array_tile.
 */
struct polysa_array_tile *polysa_array_ref_group_tile(
	struct polysa_array_ref_group *group)
{
	switch (polysa_array_ref_group_type(group)) {
	  case POLYSA_ACCESS_GLOBAL:
		  return NULL;
    case POLYSA_ACCESS_LOCAL:
      return group->local_tile;
	}
}

///* Return the effective gpu_array_tile associated to "group" or
// * NULL if there is no such polysa_array_tile.
// */
//struct polysa_array_tile *polysa_array_ref_group_tile(
//	struct polysa_array_ref_group *group)
//{
//  return group->local_tile;
//}

/* Given an index expression "index" of the form
 *
 *	L -> F(A),
 *
 * with F(A) either A or some subfield of A and L the AST loop iterators,
 * and a tiling "tiling" of the form
 *
 *	[L -> A] -> T
 *
 * apply the tiling to the outer array in the index expression to obtain
 *
 *	L -> T(A)
 *
 * If F(A) is some subfield of A, then separate the member access
 * into the base index expression and the field index expression,
 * apply the tiling to the base index expression and combine the result
 * with the field index expression.
 *
 * If F(A) is A, then modify index to keep track of the iterators
 *
 *	L -> [L -> A]
 *
 * and combine the result with the tiling to obtain a tiled index expression
 * in terms of the AST loop iterators
 *
 *	L -> T
 */
static __isl_give isl_multi_pw_aff *tile_outer(
	__isl_take isl_multi_pw_aff *index, __isl_take isl_multi_pw_aff *tiling)
{
	isl_bool is_wrapping;
	isl_space *space;
	isl_multi_pw_aff *mpa;

	is_wrapping = isl_multi_pw_aff_range_is_wrapping(index);
	if (is_wrapping < 0)
		goto error;
	if (is_wrapping) {
		isl_multi_pw_aff *field;

		field = isl_multi_pw_aff_copy(index);
		field = isl_multi_pw_aff_range_factor_range(field);
		index = isl_multi_pw_aff_range_factor_domain(index);
		index = tile_outer(index, tiling);
		return isl_multi_pw_aff_range_product(index, field);
	}

	space = isl_space_domain(isl_multi_pw_aff_get_space(index));
	space = isl_space_map_from_set(space);
	mpa = isl_multi_pw_aff_identity(space);
	index = isl_multi_pw_aff_range_product(mpa, index);
	index = isl_multi_pw_aff_pullback_multi_pw_aff(tiling, index);

	return index;
error:
	isl_multi_pw_aff_free(index);
	isl_multi_pw_aff_free(tiling);
	return NULL;
}

/* Index transformation callback for pet_stmt_build_ast_exprs.
 *
 * "index" expresses the array indices in terms of statement iterators
 *
 * We first reformulate "index" in terms of the AST loop iterators.
 * Then we check if we are accessing the global array or
 * a shared/private copy.  In particular, if we are not inside a kernel
 * then we must be accessing a global array.
 * In the former case, we simply return
 * the updated index.  If "index" is an affine expression rather
 * than an array access, then we also return the updated index here.
 *
 * If no reference groups have been computed for the array,
 * then we can only be accessing the global array.
 *
 * Otherwise, we apply the tiling to the index.
 * This tiling is of the form
 *
 *	[D -> A] -> T
 *
 * where D corresponds to the outer tile->depth dimensions of
 * the kernel schedule.
 * The index is of the form
 *
 *	L -> A
 *
 * We update the tiling to refer to the AST loop iterators
 *
 *	[L -> A] -> T
 *
 * and combine it with the index to obtain a tiled index expression in terms
 * of the AST loop iterators
 *
 *	L -> T
 *
 * Note that while the tiling applies directly to an outer array.
 * the index may refer to some subfield of this outer array.
 * In such cases, the result will refer to the same subfield of the tile.
 * That is, an index expression of the form  L -> F(A) will be transformed
 * into an index expression of the form L -> F(T).
 */
static __isl_give isl_multi_pw_aff *transform_index(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *ref_id,
	void *user)
{
	struct polysa_transform_data *data = user;
	struct polysa_stmt_access *access;
	struct polysa_array_ref_group *group;
	struct polysa_array_tile *tile;
	isl_pw_multi_aff *iterator_map;
	int i;
	int dim;
	const char *name;
	isl_space *space;
	isl_multi_pw_aff *tiling;
	isl_pw_multi_aff *pma;
	isl_pw_multi_aff *sched2depth;
  isl_pw_multi_aff *sched2copy;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_id_get_ctx(ref_id), stdout);
//  // debug

	data->array = NULL;

	iterator_map = isl_pw_multi_aff_copy(data->iterator_map);
	index = isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);

	if (!data->kernel)
		return index;

	access = find_access(data->accesses, ref_id);
	if (!access)
		return index;
	if (!isl_map_has_tuple_name(access->access, isl_dim_out)) 
		return index;

	name = get_outer_array_name(access->access);
	if (!name)
		return isl_multi_pw_aff_free(index);
	i = find_array_index(data->kernel, name);
	if (i < 0)
		isl_die(isl_multi_pw_aff_get_ctx(index), isl_error_internal,
			"cannot find array",
			return isl_multi_pw_aff_free(index));
	data->local_array = &data->kernel->array[i];
	data->array = data->local_array->array;

//  // debug
//  p = isl_printer_print_map(p, access->access);
//  printf("\n");
//  p = isl_printer_print_multi_pw_aff(p, index);
//  printf("\n");
//  // debug

	group = find_ref_group(data->local_array, access);
  data->group = group;
	if (!group) {
		data->global = 1;
    data->reg = 1;
		return index;
	}

	tile = polysa_array_ref_group_tile(group);
	data->global = !tile;
  data->reg = !tile;
	if (!tile)
		return index;

  /* recompute the sched2copy for each index */
  if (group->group_type == POLYSA_PE_GROUP) {
//    // debug
//    p = isl_printer_print_pw_multi_aff(p, data->sched2copy);
//    printf("\n");
//    // debug
//    isl_pw_multi_aff_free(data->sched2copy);
    sched2copy = compute_sched_to_copy_group(isl_pw_multi_aff_copy(data->iterator_map), group); 
//    // debug
//    p = isl_printer_print_pw_multi_aff(p, data->sched2copy);
//    printf("\n");
//    // debug
  }

	space = isl_space_domain(isl_multi_aff_get_space(tile->tiling));
	space = isl_space_range(isl_space_unwrap(space));
	space = isl_space_map_from_set(space);
	pma = isl_pw_multi_aff_identity(space);
  if (group->group_type == POLYSA_PE_GROUP) {
    sched2depth = sched2copy;
  } else {
	  sched2depth = isl_pw_multi_aff_copy(data->sched2copy);
  }
	dim = isl_pw_multi_aff_dim(sched2depth, isl_dim_out);
	sched2depth = isl_pw_multi_aff_drop_dims(sched2depth, isl_dim_out,
					    tile->depth, dim - tile->depth);
	pma = isl_pw_multi_aff_product(sched2depth, pma);
	tiling = isl_multi_pw_aff_from_multi_aff(
				    isl_multi_aff_copy(tile->tiling));
	tiling = isl_multi_pw_aff_pullback_pw_multi_aff(tiling, pma);

	index = tile_outer(index, tiling);

//  // debug
//  p = isl_printer_print_multi_pw_aff(p, index);
//  printf("\n");
//  // debug

	return index;
}

static __isl_give isl_multi_pw_aff *transform_index_module(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *ref_id,
	void *user)
{
	struct polysa_transform_data *data = user;
	struct polysa_stmt_access *access;
	struct polysa_array_ref_group *group;
	struct polysa_array_tile *tile;
	isl_pw_multi_aff *iterator_map;
	int i;
	int dim;
	const char *name;
	isl_space *space;
	isl_multi_pw_aff *tiling;
	isl_pw_multi_aff *pma;
	isl_pw_multi_aff *sched2depth;
  isl_pw_multi_aff *sched2copy;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_id_get_ctx(ref_id), stdout);
//  // debug

	data->array = NULL;

	iterator_map = isl_pw_multi_aff_copy(data->iterator_map);
	index = isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);

	if (!data->kernel)
		return index;

	access = find_access(data->accesses, ref_id);
	if (!access)
		return index;
	if (!isl_map_has_tuple_name(access->access, isl_dim_out)) 
		return index;

	name = get_outer_array_name(access->access);
	if (!name)
		return isl_multi_pw_aff_free(index);
	i = find_array_index(data->kernel, name);
	if (i < 0)
		isl_die(isl_multi_pw_aff_get_ctx(index), isl_error_internal,
			"cannot find array",
			return isl_multi_pw_aff_free(index));
	data->local_array = &data->kernel->array[i];
	data->array = data->local_array->array;

//  // debug
//  p = isl_printer_print_map(p, access->access);
//  printf("\n");
//  p = isl_printer_print_multi_pw_aff(p, index);
//  printf("\n");
//  // debug

	group = find_ref_group_module(data->local_array, access);
  data->group = group;
	if (!group) {
		data->global = 1;
    data->reg = 1;
		return index;
	}

	tile = polysa_array_ref_group_tile(group);
	data->global = !tile;
  data->reg = !tile;
	if (!tile)
		return index;

  /* recompute the sched2copy for each index */
  if (group->group_type == POLYSA_PE_GROUP) {
//    // debug
//    p = isl_printer_print_pw_multi_aff(p, data->sched2copy);
//    printf("\n");
//    // debug
//    isl_pw_multi_aff_free(data->sched2copy);
    sched2copy = compute_sched_to_copy_group(isl_pw_multi_aff_copy(data->iterator_map), group); 
//    // debug
//    p = isl_printer_print_pw_multi_aff(p, data->sched2copy);
//    printf("\n");
//    // debug
  }

	space = isl_space_domain(isl_multi_aff_get_space(tile->tiling));
	space = isl_space_range(isl_space_unwrap(space));
	space = isl_space_map_from_set(space);
	pma = isl_pw_multi_aff_identity(space);
  if (group->group_type == POLYSA_PE_GROUP) {
    sched2depth = sched2copy;
  } else {
	  sched2depth = isl_pw_multi_aff_copy(data->sched2copy);
  }
	dim = isl_pw_multi_aff_dim(sched2depth, isl_dim_out);
	sched2depth = isl_pw_multi_aff_drop_dims(sched2depth, isl_dim_out,
					    tile->depth, dim - tile->depth);
	pma = isl_pw_multi_aff_product(sched2depth, pma);
	tiling = isl_multi_pw_aff_from_multi_aff(
				    isl_multi_aff_copy(tile->tiling));
	tiling = isl_multi_pw_aff_pullback_pw_multi_aff(tiling, pma);

	index = tile_outer(index, tiling);

//  // debug
//  p = isl_printer_print_multi_pw_aff(p, index);
//  printf("\n");
//  // debug

	return index;
}

/* Dereference "expr" by adding an index [0].
 * The original "expr" is assumed not to have any indices.
 *
 * If "expr" is a member access, then the dereferencing needs
 * to be applied to the structure argument of this member access.
 */
static __isl_give isl_ast_expr *dereference(__isl_take isl_ast_expr *expr)
{
	isl_ctx *ctx;
	isl_ast_expr *arg0, *res;
	isl_ast_expr_list *list;

	arg0 = isl_ast_expr_get_op_arg(expr, 0);
	if (!arg0)
		return isl_ast_expr_free(expr);
	if (isl_ast_expr_get_type(arg0) == isl_ast_expr_op &&
	    isl_ast_expr_get_op_type(arg0) == isl_ast_op_member) {
		isl_ast_expr *arg;

		arg = isl_ast_expr_get_op_arg(arg0, 0);
		arg = dereference(arg);
		arg0 = isl_ast_expr_set_op_arg(arg0, 0, arg);
		expr = isl_ast_expr_set_op_arg(expr, 0, arg0);

		return expr;
	}
	isl_ast_expr_free(arg0);

	ctx = isl_ast_expr_get_ctx(expr);
	res = isl_ast_expr_from_val(isl_val_zero(ctx));
	list = isl_ast_expr_list_from_ast_expr(res);
	res = isl_ast_expr_get_op_arg(expr, 0);
	res = isl_ast_expr_access(res, list);
	isl_ast_expr_free(expr);

	return res;
}

/* Linearize the index expression "expr" based on the array bounds
 * of "array".
 *
 * That is, transform expression
 *
 *	A[i_0][i_1]...[i_n]
 *
 * to
 *
 *	A[(..((i_0 * b_1 + i_1) ... ) * b_n + i_n]
 *
 * where b_0, b_1, ..., b_n are the bounds on the array.
 *
 * If the base of "expr" is a member access, then the linearization needs
 * to be applied to the structure argument of this member access.
 *
 * In the base case, if "expr" has no arguments (other than the name of
 * the array), then we are passing an entire array to a function.
 * In this case, there is nothing to linearize.
 * Note that at this point an expression with no arguments can
 * only be an entire array because the scalar case and
 * the case of single struct are handled by the caller.
 *
 * If the number of specified index expressions in "expr"
 * is smaller than the dimension of the accessed array,
 * then the missing i_j also do not appear in the linearized expression.
 * Furthermore, since such an expression does not refer to a single
 * element while the default linearized expression would refer to
 * a single element, we return the expression
 *
 *	A + (..((i_0 * b_1 + i_1) ... ) * b_l + i_l)
 *
 * instead.  Note that because of the special case handling above,
 * we can assume here that there is at least one index expression.
 */
__isl_give isl_ast_expr *polysa_local_array_info_linearize_index(
	struct polysa_local_array_info *array, __isl_take isl_ast_expr *expr)
{
	int i, n;
	isl_ast_expr *arg0;
	isl_ast_expr *res;
	isl_ast_expr_list *list;

	arg0 = isl_ast_expr_get_op_arg(expr, 0);
	if (isl_ast_expr_get_type(arg0) == isl_ast_expr_op &&
	    isl_ast_expr_get_op_type(arg0) == isl_ast_op_member) {
		isl_ast_expr *arg;

		arg = isl_ast_expr_get_op_arg(arg0, 0);
		arg = polysa_local_array_info_linearize_index(array, arg);
		arg0 = isl_ast_expr_set_op_arg(arg0, 0, arg);
		expr = isl_ast_expr_set_op_arg(expr, 0, arg0);

		return expr;
	}
	isl_ast_expr_free(arg0);

	if (isl_ast_expr_get_op_n_arg(expr) == 1)
		return expr;

	n = isl_ast_expr_get_op_n_arg(expr);
	res = isl_ast_expr_get_op_arg(expr, 1);
	for (i = 1; i < array->n_index; ++i) {
		isl_ast_expr *expr_i;

		expr_i = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
		res = isl_ast_expr_mul(res, expr_i);

		if (i + 1 >= n)
			continue;
		expr_i = isl_ast_expr_get_op_arg(expr, i + 1);
		res = isl_ast_expr_add(res, expr_i);
	}

	if (1 + array->n_index > n) {
		res = isl_ast_expr_add(isl_ast_expr_get_op_arg(expr, 0), res);
	} else {
		list = isl_ast_expr_list_from_ast_expr(res);
		res = isl_ast_expr_get_op_arg(expr, 0);
		res = isl_ast_expr_access(res, list);
	}

	isl_ast_expr_free(expr);

	return res;
}

/* AST expression transformation callback for pet_stmt_build_ast_exprs.
 *
 * If the AST expression refers to an array that is not accessed
 * at all, then this means the value of the expression is not used,
 * so we might as well print zero (NULL pointer) instead.
 *
 * If the AST expression refers to a global scalar that is not
 * a read-only scalar, then its address was passed to the kernel and
 * we need to dereference it.
 *
 * If the AST expression refers to an access to a global array,
 * then we linearize the access exploiting the bounds in data->local_array.
 */
static __isl_give isl_ast_expr *transform_expr(__isl_take isl_ast_expr *expr,
	__isl_keep isl_id *id, void *user)
{
	struct polysa_transform_data *data = user;

	if (!data->array)
		return expr;

	if (!data->array->accessed) {
		isl_ctx *ctx;

		ctx = isl_ast_expr_get_ctx(expr);
		isl_ast_expr_free(expr);
		return isl_ast_expr_from_val(isl_val_zero(ctx));
	}
	if (polysa_array_is_read_only_scalar(data->array))
		return expr;
	if (!data->global)
		return expr;
	if (data->array->n_index == 0)
		return dereference(expr);
	if (!data->array->linearize)
		return expr;

	return polysa_local_array_info_linearize_index(data->local_array, expr);
}

/* AST expression transformation callback for pet_stmt_build_ast_exprs.
 *
 * If the AST expression refers to an array that is not accessed
 * at all, then this means the value of the expression is not used,
 * so we might as well print zero (NULL pointer) instead.
 *
 * If the AST expression refers to a global scalar that is not
 * a read-only scalar, then its address was passed to the kernel and
 * we need to dereference it.
 *
 * If the AST expression refers to an array reference that is put in 
 * the registers. We will modify the expr to a register access.
 *
 * If the AST expression refers to an access to a global array,
 * then we linearize the access exploiting the bounds in data->local_array.
 */
static __isl_give isl_ast_expr *transform_expr_module(__isl_take isl_ast_expr *expr,
	__isl_keep isl_id *id, void *user)
{
	struct polysa_transform_data *data = user;

	if (!data->array)
		return expr; 

	if (!data->array->accessed) {
		isl_ctx *ctx;

		ctx = isl_ast_expr_get_ctx(expr);
		isl_ast_expr_free(expr);
		return isl_ast_expr_from_val(isl_val_zero(ctx));
	}
	if (polysa_array_is_read_only_scalar(data->array))
		return expr;
//	if (!data->global)
//		return expr;
  if (!data->reg)
    return expr;
  if (data->reg) {
    isl_ctx *ctx;
    char *local_name;
    char buf[50];
    isl_id *id;
    isl_ast_expr *array;
    isl_ast_expr_list *indices;
    isl_ast_expr *indice;

    ctx = isl_ast_expr_get_ctx(expr);
//    // debug
//    isl_printer *p = isl_printer_to_file(ctx, stdout);
//    p = isl_printer_print_ast_expr(p, expr);
//    printf("\n");
//    // debug
    isl_ast_expr_free(expr);
    
    /* Create a register access. */
    isl_printer *p_str = isl_printer_to_str(ctx);    
	  p_str = polysa_array_ref_group_print_name(data->group, p_str);
    local_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    sprintf(buf, "%s", local_name);
    free(local_name);

    id = isl_id_alloc(ctx, buf, NULL);
    array = isl_ast_expr_from_id(id);

    indice = isl_ast_expr_from_val(isl_val_zero(ctx));
    indices = isl_ast_expr_list_from_ast_expr(indice);
    expr = isl_ast_expr_access(array, indices);

//    // debug
//    p = isl_printer_print_ast_expr(p, expr);
//    printf("\n");
//    // debug

    return expr;
  }
	if (data->array->n_index == 0)
		return dereference(expr);
	if (!data->array->linearize)
		return expr;

	return polysa_local_array_info_linearize_index(data->local_array, expr);
}


/* This function is called for each instance of a user statement
 * in the kernel "kernel", identified by "polysa_stmt".
 * "kernel" may be NULL if we are not inside a kernel.
 *
 * We attach a struct polysa_kernel_stmt to the "node", containing
 * a computed AST expression for each access, through an annotation
 * with name "user".
 * These AST expressions are computed from iterator_map,
 * which expresses the domain
 * elements in terms of the generated loops, and sched2copy,
 * which expresses the outer copy_schedule_dim dimensions of
 * the kernel schedule computed by PPCG in terms of the generated loops.
 */
static __isl_give isl_ast_node *create_domain_leaf(
	struct polysa_kernel *kernel, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, struct polysa_stmt *polysa_stmt)
{
	struct polysa_transform_data data;
	struct polysa_kernel_stmt *stmt;
	isl_ctx *ctx;
	isl_id *id;
	isl_pw_multi_aff *sched2copy;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	isl_union_map *schedule;

	if (!node)
		return NULL;
	ctx = isl_ast_node_get_ctx(node);

	stmt = isl_calloc_type(ctx, struct polysa_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

	schedule = isl_ast_build_get_schedule(build); 
	map = isl_map_reverse(isl_map_from_union_map(schedule));
	iterator_map = isl_pw_multi_aff_from_map(map);
	if (kernel)
		sched2copy = compute_sched_to_copy(kernel,
					isl_pw_multi_aff_copy(iterator_map)); 
	else
		sched2copy = NULL;

	stmt->type = POLYSA_KERNEL_STMT_DOMAIN;
	stmt->u.d.stmt = polysa_stmt;

	data.kernel = kernel;
	data.accesses = stmt->u.d.stmt->accesses;
	data.iterator_map = iterator_map;
	data.sched2copy = sched2copy;
	stmt->u.d.ref2expr = pet_stmt_build_ast_exprs(stmt->u.d.stmt->stmt,
					    build, &transform_index, &data,
					    &transform_expr, &data);

	isl_pw_multi_aff_free(iterator_map);
	isl_pw_multi_aff_free(sched2copy);

	id = isl_id_alloc(ctx, "user", stmt);
	id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
	if (!id)
		polysa_kernel_stmt_free(stmt);
	return isl_ast_node_set_annotation(node, id);
}

static __isl_give isl_ast_node *create_domain_leaf_module(
	struct polysa_kernel *kernel, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, struct polysa_stmt *polysa_stmt)
{
	struct polysa_transform_data data;
	struct polysa_kernel_stmt *stmt;
	isl_ctx *ctx;
	isl_id *id;
	isl_pw_multi_aff *sched2copy;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	isl_union_map *schedule;

	if (!node)
		return NULL;
	ctx = isl_ast_node_get_ctx(node);

	stmt = isl_calloc_type(ctx, struct polysa_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

	schedule = isl_ast_build_get_schedule(build); 
	map = isl_map_reverse(isl_map_from_union_map(schedule));
	iterator_map = isl_pw_multi_aff_from_map(map);
	if (kernel)
		sched2copy = compute_sched_to_copy(kernel,
					isl_pw_multi_aff_copy(iterator_map)); 
	else
		sched2copy = NULL;

	stmt->type = POLYSA_KERNEL_STMT_DOMAIN;
	stmt->u.d.stmt = polysa_stmt;

	data.kernel = kernel;
	data.accesses = stmt->u.d.stmt->accesses;
	data.iterator_map = iterator_map;
	data.sched2copy = sched2copy;
	stmt->u.d.ref2expr = pet_stmt_build_ast_exprs(stmt->u.d.stmt->stmt,
					    build, &transform_index_module, &data,
					    &transform_expr_module, &data);

	isl_pw_multi_aff_free(iterator_map);
	isl_pw_multi_aff_free(sched2copy);

	id = isl_id_alloc(ctx, "user", stmt);
	id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
	if (!id)
		polysa_kernel_stmt_free(stmt);
	return isl_ast_node_set_annotation(node, id);
}

/* Does "array" need to be allocated on the device?
 * If it is a read-only scalar, then it will be passed as an argument
 * to the kernel and therefore does not require any allocation.
 * If this device memory is not accessed at all, then it does not
 * need to be allocated either.
 */
int polysa_array_requires_device_allocation(struct polysa_array_info *array)
{
	if (polysa_array_is_read_only_scalar(array))
		return 0;
	if (!array->global)
		return 0;
	return 1;
}

/* Build AST expressions for the device array sizes of all arrays in "prog"
 * that require allocation on the device using "build", as well as
 * for the original array sizes of all arrays that need to be declared
 * on the host.
 * "node" is freed in case of error.
 */
static __isl_give isl_ast_node *build_array_bounds(
	__isl_take isl_ast_node *node, struct polysa_prog *prog,
	__isl_keep isl_ast_build *build)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];
		isl_multi_pw_aff *size;
		isl_ast_expr *expr;

		if (!polysa_array_requires_device_allocation(array))
			continue;

		size = isl_multi_pw_aff_copy(array->bound);
		expr = ppcg_build_size_expr(size, build);
		array->bound_expr = expr;
		if (!expr)
			return isl_ast_node_free(node);
	}

	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];
		isl_set *extent;
		isl_multi_pw_aff *size;
		isl_ast_expr *expr;

		if (!array->declare_local)
			continue;
		extent = isl_set_copy(array->declared_extent);
		size = ppcg_size_from_extent(extent);
		expr = ppcg_build_size_expr(size, build);
		array->declared_size = expr;
		if (!expr)
			return isl_ast_node_free(node);
	}

	return node;
}

/* Given the fifo field from the I/O statement type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[fifo_name].[is_filter].[is_buffer].[sched_depth].[param_id]
 */
static __isl_give char *extract_fifo_suffix(isl_ctx *ctx, const char *type) 
{
  char *fifo_name;
  int loc = 0;
  char ch;
  isl_printer *p_str;
  int n_dot = 0;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.')
      n_dot++;
    if (n_dot == 1)
      break;
    loc++;
  }

  p_str = isl_printer_to_str(ctx);
  loc++;
  while ((ch = type[loc]) != '\0') {
    if (ch == '.')
      break;
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  fifo_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  return fifo_name;
}

/* Extract the filter field from the I/O statement type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[fifo_name].[is_filter].[is_buffer].[sched_depth].[param_id]
 */
static int extract_is_filter(const char *type)
{
  char ch;
  int loc = 0;
  int n_dot = 0;
  int val;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.')
      n_dot++;
    if (n_dot == 2)
      break;
    loc++;
  }

  loc++;
  ch = type[loc];
  val = ch - '0';
  return val;
}

/* Extract the buffer field from the I/O statemnt type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[is_filter].[is_buffer].[fifo_name].[sched_depth].[param_id]
 */
static int extract_is_buffer(const char *type)
{
  char ch;
  int loc = 0;
  int n_dot = 0;
  int val;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.')
      n_dot++;
    if (n_dot == 3)
      break;
    loc++;
  }

  loc++;
  ch = type[loc];
  val = ch - '0';
  return val;
}

/* Extract the filter field from the I/O statemnt type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[is_filter].[is_buffer].[fifo_name].[sched_depth].[param_id]
 */
static int extract_sched_depth(isl_ctx *ctx, const char *type) 
{
  int loc = 0;
  char ch;
  int dot_time = 0;
  isl_printer *p_str;
  char *depth_str;
  int depth;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.') 
      dot_time++;
    if (dot_time == 4)
      break;
    loc++;
  }

  p_str = isl_printer_to_str(ctx);
  loc++;
  while (((ch = type[loc]) != '\0') && ((ch = type[loc]) != '.')) {
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  depth_str = isl_printer_get_str(p_str);
  depth = atoi(depth_str);
  free(depth_str);
  isl_printer_free(p_str);

  return depth;
}

/* Extract the filter field from the I/O statemnt type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[is_filter].[is_buffer].[fifo_name].[sched_depth].[param_id]
 */
static int extract_param_id(isl_ctx *ctx, const char *type)
{
  int loc = 0;
  char ch;
  int dot_time = 0;
  isl_printer *p_str;
  char *depth_str;
  int depth;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.') 
      dot_time++;
    if (dot_time == 5)
      break;
    loc++;
  }

  p_str = isl_printer_to_str(ctx);
  loc++;
  while (((ch = type[loc]) != '\0') && ((ch = type[loc]) != '.')) {
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  depth_str = isl_printer_get_str(p_str);
  depth = atoi(depth_str);
  free(depth_str);
  isl_printer_free(p_str);

  return depth;
}

/* Extract the filter field from the I/O statemnt type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[fifo_name].[is_filter].[is_buffer].[sched_depth].[param_id].[pack_lane].[nxt_pack_lane]
 * or 
 * in/out.[fifo_name].[pack_lane].[nxt_pack_lane]
 */
static int extract_data_pack(isl_ctx *ctx, const char *type, int is_trans)
{
  int loc = 0;
  char ch;
  int dot_time = 0;
  isl_printer *p_str;
  char *depth_str;
  int depth;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.') 
      dot_time++;
    if (dot_time == (is_trans? 6 : 2))
      break;
    loc++;
  }

  if (dot_time < (is_trans? 6 : 2))
    return -1;

  p_str = isl_printer_to_str(ctx);
  loc++;
  while (((ch = type[loc]) != '\0') && ((ch = type[loc]) != '.')) {
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  depth_str = isl_printer_get_str(p_str);
  depth = atoi(depth_str);
  free(depth_str);
  isl_printer_free(p_str);

  return depth; 
}

/* Extract the filter field from the I/O statemnt type.
 * The I/O statement type is in the format of:
 * in/out_trans[_dram].[fifo_name].[is_filter].[is_buffer].[sched_depth].[param_id].[pack_lane]
 * or 
 * in/out.[fifo_name].[pack_lane].[nxt_pack_lane]
 */
static int extract_next_data_pack(isl_ctx *ctx, const char *type, int is_trans) 
{
  int loc = 0;
  char ch;
  int dot_time = 0;
  isl_printer *p_str;
  char *depth_str;
  int depth;

  while ((ch = type[loc]) != '\0') {
    if (ch == '.') 
      dot_time++;
    if (dot_time == (is_trans? 7 : 3))
      break;
    loc++;
  }

  if (dot_time < (is_trans? 7 : 3))
    return -1;

  p_str = isl_printer_to_str(ctx);
  loc++;
  while (((ch = type[loc]) != '\0') && ((ch = type[loc]) != '.')) {
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  depth_str = isl_printer_get_str(p_str);
  depth = atoi(depth_str);
  free(depth_str);
  isl_printer_free(p_str);

  return depth; 
}

__isl_give isl_ast_expr_list *decl_args_generate(struct polysa_hw_module *module)
{
  isl_ast_expr_list *list;
  struct polysa_kernel *kernel;
  struct polysa_prog *prog;
  isl_ctx *ctx;

  kernel = module->kernel;
  prog = kernel->prog;
  ctx = prog->ctx; 
  list = isl_ast_expr_list_alloc(ctx ,0);

  /* Array */
  if (module->type != PE_MODULE && module->level == 3) {
    isl_id *id;
    isl_ast_expr *expr;

    id = isl_id_alloc(ctx, module->io_groups[0]->array->name, NULL);
    expr = isl_ast_expr_from_id(id);

    list = isl_ast_expr_list_add(list, expr);
  } else if (module->type == PE_MODULE) {
    /* Scalars */
    for (int i = 0; i < prog->n_array; i++) {
      struct polysa_array_info *array = &prog->array[i];
      int required;

      required = polysa_kernel_requires_array_argument(kernel, i);
      if (!required)
        continue;

      if (polysa_array_is_read_only_scalar(array)) {
        isl_id *id;
        isl_ast_expr *expr;

        id = isl_id_alloc(ctx, array->name, NULL);
        expr = isl_ast_expr_from_id(id);

        list = isl_ast_expr_list_add(list, expr);
      }
    }
  }

  /* fifos */
  if (module->type == PE_MODULE) {
    for (int i = 0; i < module->n_io_group; i++) {
      if (module->io_groups[i]->pe_io_dir == IO_IN ||
          module->io_groups[i]->pe_io_dir == IO_INOUT) {
        struct polysa_array_ref_group *group = module->io_groups[i];
        isl_printer *p_str;

        p_str = isl_printer_to_str(ctx);
        p_str = polysa_array_ref_group_print_fifo_name(group, p_str);
      }
    }
  } else {

  }

  return list;
}

static __isl_give isl_ast_node *create_module_call_leaf(struct polysa_kernel *kernel,
  __isl_take isl_ast_node *node, struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, const char *name, __isl_keep isl_ast_build *build)
{
  struct polysa_kernel_stmt *stmt;
  isl_id *id;
  isl_ctx *ctx;
  isl_multi_aff *trans;
  isl_map *map;
  isl_pw_multi_aff *pma;
  isl_ast_expr *expr;

  ctx = isl_ast_node_get_ctx(node);
  stmt = isl_calloc_type(ctx, struct polysa_kernel_stmt);
  if (!stmt)
    return isl_ast_node_free(node);
 
  stmt->type = POLYSA_KERNEL_STMT_MODULE_CALL;
  stmt->u.m.module = module;
  stmt->u.m.group = group;
  if (!prefixcmp(name, "module_call_lower")) {
    stmt->u.m.lower = 1;
    stmt->u.m.upper = 0;
  } else if (!prefixcmp(name, "module_call_upper")) {
    stmt->u.m.lower = 0;
    stmt->u.m.upper = 1;
  } else {
    stmt->u.m.lower = 0;
    stmt->u.m.upper = 0;
  }

  if (stmt->u.m.lower) {
    if ((module->type == IO_MODULE || module->type == DRAIN_MODULE) && !group->io_pe_expr) {
      if (module->to_pe) {
//        // debug
//        isl_printer *p = isl_printer_to_file(isl_ast_node_get_ctx(node), stdout);
//        // debug
        isl_union_map *umap = isl_ast_build_get_schedule(build);
        isl_union_set *uset = isl_union_map_range(umap);
        isl_set *set = isl_set_from_union_set(uset);
        isl_map *map = isl_set_identity(set);
        map = isl_map_flatten_range(map);
        trans = isl_multi_aff_copy(group->io_trans);
        isl_map *map2 = isl_map_from_multi_aff(trans);
        map2 = isl_map_reverse(map2);
//        // debug
//        p = isl_printer_print_map(p, map);
//        printf("\n");
//        p = isl_printer_print_map(p, map2);
//        printf("\n");
//        // debug
        map = isl_map_apply_range(map, map2);
        isl_pw_multi_aff *pma = isl_pw_multi_aff_from_map(map);
        expr = isl_ast_build_access_from_pw_multi_aff(build, pma);
        group->io_pe_expr = expr;
      }
    }
  }

  id = isl_id_alloc(ctx, "module_call", stmt);
  id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
  if (!id)
    polysa_kernel_stmt_free(stmt);
  return isl_ast_node_set_annotation(node, id);
}

static __isl_give isl_ast_node *create_fifo_decl_leaf(struct polysa_kernel *kernel,
  __isl_take isl_ast_node *node, struct polysa_hw_module *module,
  struct polysa_array_ref_group *group, const char *name, __isl_keep isl_ast_build *build)
{
  struct polysa_kernel_stmt *stmt;
  isl_id *id;
  isl_ctx *ctx;
  isl_multi_aff *trans;
  isl_map *map;
  isl_pw_multi_aff *pma;
  isl_ast_expr *expr;

  ctx = isl_ast_node_get_ctx(node);
  stmt = isl_calloc_type(ctx, struct polysa_kernel_stmt);
  if (!stmt)
    return isl_ast_node_free(node);

  /* Generate the ast expr of io_trans */ 
  if (module->type == PE_MODULE && !group->io_L1_pe_expr) {
    isl_union_map *umap = isl_ast_build_get_schedule(build);
    isl_union_set *uset = isl_union_map_range(umap);
    isl_set *set = isl_set_from_union_set(uset);
    isl_map *map = isl_set_identity(set);
    map = isl_map_flatten_range(map);
    trans = group->io_L1_trans;
    isl_map *map2 = isl_map_from_multi_aff(isl_multi_aff_copy(trans));
    map2 = isl_map_reverse(map2);
    map = isl_map_apply_range(map, map2);
    isl_pw_multi_aff *pma = isl_pw_multi_aff_from_map(map);
    expr = isl_ast_build_access_from_pw_multi_aff(build, pma);
    group->io_L1_pe_expr = expr;
  } 

  stmt->type = POLYSA_KERNEL_STMT_FIFO_DECL;
  stmt->u.m.module = module;
  stmt->u.m.group = group;
  if (!prefixcmp(name, "fifo_decl_boundary"))
    stmt->u.m.boundary = 1;
  else
    stmt->u.m.boundary = 0;
  id = isl_id_alloc(ctx, "fifo_decl", stmt);
  id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
  if (!id)
    polysa_kernel_stmt_free(stmt);
  return isl_ast_node_set_annotation(node, id);
}

static __isl_give isl_ast_node *create_io_module_call_leaf(struct polysa_kernel *kernel,
  __isl_take isl_ast_node *node, struct polysa_hw_module *module,
  const char *name, __isl_keep isl_ast_build *build)
{
  isl_id *id;
  isl_ctx *ctx;
  struct polysa_kernel_stmt *stmt;
  
  ctx = isl_ast_node_get_ctx(node);
  stmt = isl_calloc_type(ctx, struct polysa_kernel_stmt);
  if (!stmt)
    return isl_ast_node_free(node);

  stmt->u.f.module = module;
  if (!strcmp(name, "io_module.inter_trans"))
    stmt->type = POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_TRANS;
  else if (!strcmp(name, "io_module.intra_trans"))
    stmt->type = POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_TRANS;
  else if (!strcmp(name, "io_module.inter_intra"))
    stmt->type = POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_INTRA;
  else if (!strcmp(name, "io_module.intra_inter"))
    stmt->type = POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_INTER;
  else if (!strcmp(name, "io_module.state_handle"))
    stmt->type = POLYSA_KERNEL_STMT_IO_MODULE_CALL_STATE_HANDLE;
  id = isl_id_alloc(ctx, name, stmt);
  id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
  if (!id)
    polysa_kernel_stmt_free(stmt);
  return isl_ast_node_set_annotation(node, id);
}

/* This function is called for each statement node in the AST
 * for transferring through fifos.
 * Attach a pointer to a polysa_kernel_stmt representing the io
 * statemet to the node.
 * The statement name is "in" or "out", depending on whether we are 
 * transferring in or out via fifos.
 *
 * The schedule is of the form
 *
 *  type[D -> A] -> L
 *
 * where D corresponds to the outer tile->depth dimensions of 
 * the kernel schedule, A to the global array and L to the outer 
 * generated AST schedule.
 * We compute the inverse and strip off the type, resulting in
 *
 *  L -> [D -> A]
 *
 * We combine this mapping with the group tiling
 *
 *  [D -> A] -> T
 *
 * resulting in
 *   
 *  L -> T
 *
 * and store the corresponding expressions in stmt->local_index,
 * where stmt points to the ppcg_kernel_stmt that is attached to the node.
 */
static __isl_give isl_ast_node *create_io_leaf(struct polysa_kernel *kernel,
  struct polysa_hw_module *module, struct polysa_array_ref_group_pair *pair, 
  __isl_take isl_ast_node *node, __isl_keep isl_ast_build *build)
{
  struct polysa_kernel_stmt *stmt;
  struct polysa_array_tile *tile;
  isl_multi_aff *new_tiling;
  isl_map *access;
  const char *type;
  isl_pw_multi_aff *pma, *pma2;
  isl_space *space;
  isl_ast_expr *expr;
  isl_id *id;
  int is_trans; // i/o transfer statment betwen on-chip modules
  int is_trans_dram; // i/o transfer statement betwen dram and on-chip modules
  int is_trans_filter; // i/o transfer statment with filters
  int is_trans_buf; // i/o transfer statment with local buffers
  struct polysa_array_ref_group *group = pair->local_group;
  int depth;
  isl_ctx *ctx;

  stmt = isl_calloc_type(kernel->ctx, struct polysa_kernel_stmt); 
  if (!stmt)
    return isl_ast_node_free(node);

  ctx = kernel->ctx;

//  // debug
//  isl_printer *p = isl_printer_to_file(kernel->ctx, stdout);
//  // debug

  /* type[D -> A] -> L */
  access = isl_map_from_union_map(isl_ast_build_get_schedule(build)); 
  isl_set *set = isl_map_domain(isl_set_unwrap(isl_map_domain(isl_map_copy(access)))); 
  depth = isl_set_dim(set, isl_dim_set);
  isl_set_free(set);

  type = isl_map_get_tuple_name(access, isl_dim_in);
  /* Classify the io stmt type */
  is_trans = !prefixcmp(type, "in_trans") || !prefixcmp(type, "out_trans");
  is_trans_dram = !prefixcmp(type, "in_trans_dram") || !prefixcmp(type, "out_trans_dram");
  if (is_trans || is_trans_dram) {
    is_trans_filter = extract_is_filter(type);
    is_trans_buf = extract_is_buffer(type);
  }
  stmt->u.i.in = type && !prefixcmp(type, "in");
  stmt->u.i.buf = is_trans_buf;
  stmt->u.i.filter = is_trans_filter;
  stmt->u.i.data_pack = extract_data_pack(ctx, type, is_trans || is_trans_dram);
  stmt->u.i.nxt_data_pack = extract_next_data_pack(ctx, type, is_trans || is_trans_dram);
    
  /* Compute the global index */
  /* L -> type[D -> A] */
  access = isl_map_reverse(access); 
  pma = isl_pw_multi_aff_from_map(access); 
  pma = isl_pw_multi_aff_reset_tuple_id(pma, isl_dim_out); 

  space = isl_space_range(isl_pw_multi_aff_get_space(pma)); 
  space = isl_space_unwrap(space);
  /* [D -> A] -> A */
  pma2 = isl_pw_multi_aff_range_map(space); 
  /* L -> A */
  pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2,
            isl_pw_multi_aff_copy(pma));
  expr = isl_ast_build_access_from_pw_multi_aff(build, pma2); 
  if (group->array->linearize) {
    expr = polysa_local_array_info_linearize_index(group->local_array,
              expr);

    if (stmt->u.i.data_pack > 1) {
      isl_ast_expr *arg, *div;
      arg = isl_ast_expr_get_op_arg(expr, 1);
      div = isl_ast_expr_from_val(isl_val_int_from_si(kernel->ctx, stmt->u.i.data_pack));
      arg = isl_ast_expr_div(arg, div); 
      expr = isl_ast_expr_set_op_arg(expr, 1, arg);
    }
  } else {
    if (stmt->u.i.data_pack > 1) {
      int n_arg;
      isl_ast_expr *arg, *div;
      n_arg = isl_ast_expr_get_op_n_arg(expr);
      arg = isl_ast_expr_get_op_arg(expr, n_arg - 1);
      div = isl_ast_expr_from_val(isl_val_int_from_si(kernel->ctx, stmt->u.i.data_pack));
      arg = isl_ast_expr_div(arg, div);
      expr = isl_ast_expr_set_op_arg(expr, n_arg - 1, arg);
    }
  }
//  // debug
//  p = isl_printer_print_ast_expr(p, expr);
//  printf("\n");
//  // debug

  stmt->u.i.index = expr; 

  /* Compute the local index */
  tile = pair->local_tile;
  if (tile) {
    isl_ast_expr *arg, *div;
    int n_arg;

    /* [D -> A] -> T */
    pma2 = isl_pw_multi_aff_from_multi_aff(
              isl_multi_aff_copy(tile->tiling)); 
    if (tile->depth < depth) {
      /* Extend the D dimension to depth in pma2 */
      new_tiling = polysa_array_ref_group_recompute_tiling(tile, group, depth); 
      isl_pw_multi_aff_free(pma2);
      pma2 = isl_pw_multi_aff_from_multi_aff(new_tiling);
    }
    
    /* L -> T */
    pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2, pma);
    expr = isl_ast_build_access_from_pw_multi_aff(build, pma2);
//    // debug
//    p = isl_printer_print_ast_expr(p, expr);
//    printf("\n");
//    // debug

//    if (stmt->u.i.data_pack > 1) {
//      n_arg = isl_ast_expr_get_op_n_arg(expr);
//      arg = isl_ast_expr_get_op_arg(expr, n_arg - 1);
//      div = isl_ast_expr_from_val(isl_val_int_from_si(kernel->ctx, stmt->u.i.data_pack));
//      arg = isl_ast_expr_div(arg, div);
//      expr = isl_ast_expr_set_op_arg(expr, n_arg - 1, arg);
//    }

//    // debug
//    p = isl_printer_print_ast_expr(p, expr);
//    printf("\n");
//    // debug
    stmt->u.i.local_index = expr;
    stmt->u.i.reg = 0;
  } else {
    /* Create a scalar expr */
    isl_printer *p_str;
    char *local_name;
    char buf[50];
    isl_ast_expr *array, *indice;
    isl_ast_expr_list *indices;

    isl_pw_multi_aff_free(pma);
    p_str = isl_printer_to_str(kernel->ctx);
    p_str = polysa_array_ref_group_print_name(group, p_str);
    local_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
    sprintf(buf, "%s", local_name);
    free(local_name);

    id = isl_id_alloc(kernel->ctx, buf, NULL);
    array = isl_ast_expr_from_id(id);
    indice = isl_ast_expr_from_val(isl_val_zero(kernel->ctx));
    indices = isl_ast_expr_list_from_ast_expr(indice);
    expr = isl_ast_expr_access(array, indices);
//    // debug
//    p = isl_printer_print_ast_expr(p, expr);
//    printf("\n");
//    // debug
    stmt->u.i.local_index = expr;
    stmt->u.i.reg = 1;
  }

  isl_printer *p_str = isl_printer_to_str(isl_ast_node_get_ctx(node));
  char *fifo_name = extract_fifo_suffix(ctx, type);
  p_str = isl_printer_print_str(p_str, fifo_name);
  free(fifo_name);
  stmt->u.i.fifo_name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  
  stmt->u.i.group = pair->io_group;
  stmt->u.i.module = module;
  stmt->u.i.array = group->array;
  stmt->u.i.local_array = group->local_array;   
  if (is_trans) {
    if (is_trans_dram) {
      stmt->type = POLYSA_KERNEL_STMT_IO_DRAM;
    } else {      
      stmt->type = POLYSA_KERNEL_STMT_IO_TRANSFER;
      if (is_trans_filter) {
        stmt->u.i.filter_sched_depth = extract_sched_depth(ctx, type);
        stmt->u.i.filter_param_id = extract_param_id(ctx, type);
      } else {
        stmt->u.i.filter_sched_depth = -1;
        stmt->u.i.filter_param_id = -1;
      }
    }
  } else {
    stmt->type = POLYSA_KERNEL_STMT_IO;
  }

  id = isl_id_alloc(kernel->ctx, "io", stmt);
  id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
  if (!id)
    polysa_kernel_stmt_free(stmt);
  return isl_ast_node_set_annotation(node, id);
}

/* This function is called for each statement node in the AST
 * for copying to or from local memory.
 * Attach a pointer to a polysa_kernel_stmt representing the copy
 * statement to the node.
 * The statement name is "read" or "write", depending on whether we are
 * reading from global memory or writing to global memory.
 *
 * The schedule is of the form
 *
 *	type[D -> A] -> L
 *
 * where D corresponds to the outer tile->depth dimensions of
 * the kernel schedule, A to the global array and L to the outer
 * generated AST schedule.
 * We compute the inverse and strip off the type, resulting in
 *
 *	L -> [D -> A]
 *
 * We combine this mapping with on the one hand the projection
 *
 *	[D -> A] -> A
 *
 * and on the other hand the group tiling
 *
 *	[D -> A] -> T
 *
 * resulting in
 *
 *	L -> A		and 	L -> T
 *
 * and store the corresponding expressions in stmt->index and stmt->local_index,
 * where stmt points to the ppcg_kernel_stmt that is attached to the node.
 * stmt->index is linearized if the global memory array is linearized.
 */
static __isl_give isl_ast_node *create_access_leaf(struct polysa_kernel *kernel,
	struct polysa_array_ref_group *group, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build)
{
	struct polysa_kernel_stmt *stmt;
	struct polysa_array_tile *tile;
	isl_id *id;
	isl_ast_expr *expr;
	isl_space *space;
	isl_map *access;
	isl_pw_multi_aff *pma, *pma2;
	const char *type;

	stmt = isl_calloc_type(kernel->ctx, struct polysa_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

  /* type[D -> A] -> L */
	access = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	type = isl_map_get_tuple_name(access, isl_dim_in);
	stmt->u.c.read = type && !strcmp(type, "read");
  /* L -> type[D -> A] */
	access = isl_map_reverse(access);
	pma = isl_pw_multi_aff_from_map(access);
	pma = isl_pw_multi_aff_reset_tuple_id(pma, isl_dim_out);

	space = isl_space_range(isl_pw_multi_aff_get_space(pma));
	space = isl_space_unwrap(space);
  /* [D -> A] -> A */
	pma2 = isl_pw_multi_aff_range_map(space);
  /* L -> A */
	pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2,
						    isl_pw_multi_aff_copy(pma));
	expr = isl_ast_build_access_from_pw_multi_aff(build, pma2);
	if (group->array->linearize)
		expr = polysa_local_array_info_linearize_index(group->local_array,
							    expr);
	stmt->u.c.index = expr;

	tile = polysa_array_ref_group_tile(group);
  /* [D -> A] -> T */
	pma2 = isl_pw_multi_aff_from_multi_aff(
					    isl_multi_aff_copy(tile->tiling));
  /* L -> T */
	pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2, pma);
	expr = isl_ast_build_access_from_pw_multi_aff(build, pma2);
	stmt->u.c.local_index = expr;

	stmt->u.c.array = group->array;
	stmt->u.c.local_array = group->local_array;
	stmt->type = POLYSA_KERNEL_STMT_COPY;

	id = isl_id_alloc(kernel->ctx, "copy", stmt);
	id = isl_id_set_free_user(id, &polysa_kernel_stmt_free);
	if (!id)
		polysa_kernel_stmt_free(stmt);
	return isl_ast_node_set_annotation(node, id);
}

/* This function is called for each instance of a user statement
 * in the kernel.  This may be one of the original user statements
 * or a statement introduced by PolySA.
 *
 * We first check if the statement id corresponds to a fpga statement,
 * which indicates the statement is an original user statement. Any statement
 * that is not an original user statement has been introduced by PolySA and
 * requires special handling.
 *
 * If the user statement is one of the original user statements, then we call
 * create_domain_leaf.  If it is "init_device", then we call
 * build_array_bounds.  
 * Otherwise, we check if it is a copy
 * statement and call the appropriate functions.  Statements that copy an array
 * to/from the device do not need any further treatment.
 * Neither does "clear_device".
 */
static __isl_give isl_ast_node *at_domain(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct polysa_at_domain_data *data = user;
	struct polysa_stmt *device_stmt;
	isl_ast_expr *expr, *arg;
	isl_id *id;
	int is_sync;
	const char *name;
	void *p;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	p = isl_id_get_user(id);
	isl_ast_expr_free(expr);
	isl_ast_expr_free(arg);

	device_stmt = find_stmt(data->prog, id);
  isl_id_free(id);

  if (device_stmt)
    return create_domain_leaf(data->kernel, node, build, device_stmt); 

  if (!prefixcmp(name, "to_device_") || !prefixcmp(name, "from_device_"))
    return node;
  if (!strcmp(name, "init_device"))
    return build_array_bounds(node, data->prog, build); 
  if (!strcmp(name, "clear_device"))
    return node;
  if (!strcmp(name, "read") || !strcmp(name, "write")) {
    struct polysa_array_ref_group *group = p;
    return create_access_leaf(data->kernel, group, node, build);
  }

  return node;
}

static __isl_give isl_ast_node *at_domain_module(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct polysa_at_domain_data *data = user;
	struct polysa_stmt *device_stmt;
	isl_ast_expr *expr, *arg;
	isl_id *id;
	int is_sync;
	const char *name;
	void *p;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	p = isl_id_get_user(id);
	isl_ast_expr_free(expr);
	isl_ast_expr_free(arg);

	device_stmt = find_stmt(data->prog, id);
  isl_id_free(id);

  if (device_stmt)
    return create_domain_leaf_module(data->kernel, node, build, device_stmt); 

  if (!prefixcmp(name, "to_device_") || !prefixcmp(name, "from_device_"))
    return node;
  if (!strcmp(name, "init_device"))
    return build_array_bounds(node, data->prog, build); 
  if (!strcmp(name, "clear_device"))
    return node;
  if (!strcmp(name, "read") || !strcmp(name, "write")) {
    struct polysa_array_ref_group *group = p;
    return create_access_leaf(data->kernel, group, node, build);
  }
  if (!prefixcmp(name, "in") || !prefixcmp(name, "out")) {
    struct polysa_array_ref_group_pair *pair = p;
    return create_io_leaf(data->kernel, data->module, pair, node, build);
  }
  if (!prefixcmp(name, "module_call")) {
    /* module_call.[module_name]
     * module_call_lower.[module_name]
     */
    struct polysa_array_ref_group *group = NULL;
    if (!prefixcmp(name, "module_call_lower"))
      group = p;
    return create_module_call_leaf(data->kernel, node, data->module, group, name, build);
  }
  if (!prefixcmp(name, "fifo_decl")) {
    /* fifo_decl.[fifo_name]
     * fifo_decl_boundary.[fifo_name]
     */
    struct polysa_array_ref_group *group = p;
    return create_fifo_decl_leaf(data->kernel, node, data->module, group, name, build);
  }
  if (!prefixcmp(name, "io_module")) {
    return create_io_module_call_leaf(data->kernel, node, data->module, name, build);
  }

  return node;
}


///* Build an access AST expression for the effective grid size using "build".
// * Store the result in kernel->grid_size_expr.
// */
//static isl_stat build_grid_size(struct ppcg_kernel *kernel,
//	__isl_keep isl_ast_build *build)
//{
//	isl_multi_pw_aff *size;
//
//	size = isl_multi_pw_aff_copy(kernel->grid_size);
//	size = isl_multi_pw_aff_set_tuple_name(size, isl_dim_out, "grid");
//	kernel->grid_size_expr = ppcg_build_size_expr(size, build);
//
//	if (!kernel->grid_size_expr)
//		return isl_stat_error;
//	return isl_stat_ok;
//}

/* Build access AST expressions for the localized array sizes using "build".
 * Store the result in local->bound_expr.
 * Only do this for arrays for which localized bounds have been computed.
 */
static isl_stat build_local_array_sizes(struct polysa_kernel *kernel,
	__isl_keep isl_ast_build *build)
{
	int i;

	for (i = 0; i < kernel->n_array; ++i) {
		struct polysa_local_array_info *local = &kernel->array[i];
		isl_multi_pw_aff *size;

		if (local->n_group == 0)
			continue;
		size = isl_multi_pw_aff_copy(local->bound);
		local->bound_expr = ppcg_build_size_expr(size, build);
		if (!local->bound_expr)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Build an access AST expression for the effective grid size using "build".
 * Store the result in kernel->grid_size_expr.
 */
static isl_stat build_grid_size(struct polysa_kernel *kernel,
	__isl_keep isl_ast_build *build)
{
	isl_multi_pw_aff *size;

	size = isl_multi_pw_aff_copy(kernel->grid_size);
	size = isl_multi_pw_aff_set_tuple_name(size, isl_dim_out, "grid");
	kernel->grid_size_expr = ppcg_build_size_expr(size, build);

	if (!kernel->grid_size_expr)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Build access AST expressions for the effective grid size and
 * the localized array sizes using "build".
 */
static isl_stat build_grid_and_local_array_sizes(struct polysa_kernel *kernel,
	__isl_keep isl_ast_build *build)
{
	if (build_grid_size(kernel, build) < 0)
		return isl_stat_error;
	if (build_local_array_sizes(kernel, build) < 0)
		return isl_stat_error;
	return isl_stat_ok;
}

/* This function is called before the AST generator starts traversing
 * the schedule subtree of a node with mark "mark".
 *
 * If the mark is called "kernel", store the kernel pointer in data->kernel
 * for use in at_domain and build AST expressions for the grid size and
 * the localized array sizes.
 */
static isl_stat before_mark(__isl_keep isl_id *mark,
	__isl_keep isl_ast_build *build, void *user)
{
	struct polysa_at_domain_data *data = user;

	if (!mark)
		return isl_stat_error;
	if (!strcmp(isl_id_get_name(mark), "kernel")) {
		data->kernel = isl_id_get_user(mark);
		if (build_grid_and_local_array_sizes(data->kernel, build) < 0)
			return isl_stat_error;
	}
	return isl_stat_ok;
}

/* This function is called after the AST generator has finished traversing
 * the schedule subtree of a mark node.  "node" points to the corresponding
 * mark AST node.
 *
 * If the mark is called "kernel", then replace "node" by a user node
 * that "calls" the kernel, representing the launch of the kernel.
 * The original "node" is stored inside the kernel object so that
 * it can be used to print the device code.
 * Note that this assumes that a kernel is only launched once.
 * Also clear data->kernel.
 */
static __isl_give isl_ast_node *after_mark(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_ast_expr *expr;
	isl_ast_expr_list *list;
	struct polysa_kernel *kernel;
	struct polysa_at_domain_data *data = user;

	ctx = isl_ast_node_get_ctx(node);
	id = isl_ast_node_mark_get_id(node);
	if (!id)
		return isl_ast_node_free(node);
	if (strcmp(isl_id_get_name(id), "kernel") || !data->kernel) {
		isl_id_free(id);
		return node;
	}
	kernel = data->kernel;
	data->kernel = NULL;
	kernel->space = isl_ast_build_get_schedule_space(build);
	kernel->tree = isl_ast_node_mark_get_node(node);
	isl_ast_node_free(node);

//  // debug
//  isl_printer *p_d = isl_printer_to_file(isl_ast_node_get_ctx(kernel->tree), stdout);
//  p_d = isl_printer_set_output_format(p_d, ISL_FORMAT_C);
//  p_d = isl_printer_print_ast_node(p_d, kernel->tree);
//  printf("\n");
//  // debug

	expr = isl_ast_expr_from_id(isl_id_copy(id));
	list = isl_ast_expr_list_alloc(ctx, 0);
	expr = isl_ast_expr_call(expr, list);
	node = isl_ast_node_alloc_user(expr);
	node = isl_ast_node_set_annotation(node, id);

	return node;
}

/* This function is called before the AST generator starts traversing
 * the schedule subtree of a node with mark "mark".
 *
 * If the mark is called "kernel", store the kernel pointer in data->kernel
 * for use in at_domain_module.
 * If the mark is called "module", store the kernel pointer in data->module
 * for use in at_domain_module.
 */
static isl_stat before_mark_module(__isl_keep isl_id *mark,
	__isl_keep isl_ast_build *build, void *user)
{
	struct polysa_at_domain_data *data = user;

	if (!mark)
		return isl_stat_error;
  if (!strcmp(isl_id_get_name(mark), "kernel")) {
    data->kernel = isl_id_get_user(mark);
  }
	if (!strcmp(isl_id_get_name(mark), "module")) {
		data->module = isl_id_get_user(mark);
	}
  if (!strcmp(isl_id_get_name(mark), "io_module.inter_trans") ||
      !strcmp(isl_id_get_name(mark), "io_module.intra_trans")) {
    data->filter_buffer = 1;
  }
  if (!strcmp(isl_id_get_name(mark), "hls_pipeline")) {
    data->under_pipeline = 1;
  }
  if (!strcmp(isl_id_get_name(mark), "hls_unroll")) {
    data->under_unroll = 1;
  }

	return isl_stat_ok;
}

/* This function is called after the AST generator has finished traversing
 * the schedule subtree of a mark node.  "node" points to the corresponding
 * mark AST node.
 *
 * If the mark is called "module", then replace "node" by a user node
 * that "calls" the module, representing the launch of the module.
 * The original "node" is stored inside the module object so that
 * it can be used to print the device code.
 * Also clear data->module.
 */
static __isl_give isl_ast_node *after_mark_module(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_ast_expr *expr;
	isl_ast_expr_list *list;
	struct polysa_kernel *kernel;
	struct polysa_at_domain_data *data = user;
  struct polysa_hw_module *module;

	ctx = isl_ast_node_get_ctx(node);
	id = isl_ast_node_mark_get_id(node);
	if (!id)
		return isl_ast_node_free(node);

  if (!strcmp(isl_id_get_name(id), "kernel") && data->kernel) {
    isl_id_free(id);
    if (!data->kernel->space)
      data->kernel->space = isl_ast_build_get_schedule_space(build);
    data->kernel = NULL;
    return node;
  }
  if (!strcmp(isl_id_get_name(id), "io_module.inter_trans")) {
    module = data->module;
    if (!module->inter_space)
      module->inter_space = isl_ast_build_get_schedule_space(build);

    module->inter_tree = isl_ast_node_mark_get_node(node);
    isl_ast_node_free(node);
    
    expr = isl_ast_expr_from_id(isl_id_copy(id));
    list = isl_ast_expr_list_alloc(ctx, 0);
    expr = isl_ast_expr_call(expr, list);
    node = isl_ast_node_alloc_user(expr);
    node = isl_ast_node_set_annotation(node, id);
    
//    data->filter_buffer = 0;

    return node;
  }
  if (!strcmp(isl_id_get_name(id), "io_module.intra_trans")) {
    module = data->module;
    if (!module->intra_space)
      module->intra_space = isl_ast_build_get_schedule_space(build);

    module->intra_tree = isl_ast_node_mark_get_node(node);
    isl_ast_node_free(node);

    expr = isl_ast_expr_from_id(isl_id_copy(id));
    list = isl_ast_expr_list_alloc(ctx, 0);
    expr = isl_ast_expr_call(expr, list);
    node = isl_ast_node_alloc_user(expr);
    node = isl_ast_node_set_annotation(node, id);

//    data->filter_buffer = 0;

    return node;
  }
  if (!strcmp(isl_id_get_name(id), "hls_pipeline")) {
    isl_id_free(id);
    data->under_pipeline = 0;

    return node;
  }
  if (!strcmp(isl_id_get_name(id), "hls_unroll")) {
    isl_id_free(id);
    data->under_unroll = 0;

    return node;
  }

	if (strcmp(isl_id_get_name(id), "module") || !data->module) {
		isl_id_free(id);
		return node;
	}

  if (data->filter_buffer == 0) {
    module = data->module;
    data->module = NULL;
    module->device_tree = isl_ast_node_mark_get_node(node);
	  isl_ast_node_free(node);
    if (!module->space)
      module->space = isl_ast_build_get_schedule_space(build);

	  expr = isl_ast_expr_from_id(isl_id_copy(id));
	  list = isl_ast_expr_list_alloc(ctx, 0);
	  expr = isl_ast_expr_call(expr, list);
	  node = isl_ast_node_alloc_user(expr);
	  node = isl_ast_node_set_annotation(node, isl_id_copy(id));
  }
  isl_id_free(id);

	return node;
}

static __isl_give isl_ast_node *after_mark_module_call(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_ast_expr *expr;
	isl_ast_expr_list *list;
	struct polysa_kernel *kernel;
	struct polysa_at_domain_data *data = user;
  struct polysa_hw_module *module;
  struct polysa_hw_top_module *top;

	ctx = isl_ast_node_get_ctx(node);
	id = isl_ast_node_mark_get_id(node);
	if (!id)
		return isl_ast_node_free(node);

  if (!strcmp(isl_id_get_name(id), "kernel") && data->kernel) {
    isl_id_free(id);
    if (!data->kernel->space)
      data->kernel->space = isl_ast_build_get_schedule_space(build);
    data->kernel = NULL;
    return node;
  }
	if (strcmp(isl_id_get_name(id), "module") || !data->module) {
		isl_id_free(id);
		return node;
	}
  top = data->top;
  data->top = NULL;
  top->n_module_call_wrapped++;
  top->module_call_wrapped_trees = (isl_ast_node **)realloc(top->module_call_wrapped_trees,
    top->n_module_call_wrapped * sizeof(isl_ast_node *));
  top->module_call_wrapped_trees[top->n_module_call_wrapped - 1] = 
    isl_ast_node_mark_get_node(node);
	isl_ast_node_free(node);

	expr = isl_ast_expr_from_id(isl_id_copy(id));
	list = isl_ast_expr_list_alloc(ctx, 0);
	expr = isl_ast_expr_call(expr, list);
	node = isl_ast_node_alloc_user(expr);
	node = isl_ast_node_set_annotation(node, id);

	return node;
}

static __isl_give isl_ast_node *after_mark_fifo_decl(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_ast_expr *expr;
	isl_ast_expr_list *list;
	struct polysa_kernel *kernel;
	struct polysa_at_domain_data *data = user;
  struct polysa_hw_module *module;
  struct polysa_hw_top_module *top;

	ctx = isl_ast_node_get_ctx(node);
	id = isl_ast_node_mark_get_id(node);
	if (!id)
		return isl_ast_node_free(node);

  if (!strcmp(isl_id_get_name(id), "kernel") && data->kernel) {
    isl_id_free(id);
    if (!data->kernel->space)
      data->kernel->space = isl_ast_build_get_schedule_space(build);
    data->kernel = NULL;
    return node;
  }
	if (strcmp(isl_id_get_name(id), "module") || !data->module) {
		isl_id_free(id);
		return node;
	}
  top = data->top;
  data->top = NULL;
  top->n_fifo_decl_wrapped++;
  top->fifo_decl_wrapped_trees = (isl_ast_node **)realloc(top->fifo_decl_wrapped_trees,
    top->n_fifo_decl_wrapped * sizeof(isl_ast_node *));
  top->fifo_decl_wrapped_trees[top->n_fifo_decl_wrapped - 1] = 
    isl_ast_node_mark_get_node(node);
	isl_ast_node_free(node);

	expr = isl_ast_expr_from_id(isl_id_copy(id));
	list = isl_ast_expr_list_alloc(ctx, 0);
	expr = isl_ast_expr_call(expr, list);
	node = isl_ast_node_alloc_user(expr);
	node = isl_ast_node_set_annotation(node, id);

	return node;
}

static void free_ast_node_userinfo(void *ptr)
{
  struct polysa_ast_node_userinfo *info = ptr;
  free(info);
}

static __isl_give isl_id *before_for_module(
  __isl_keep isl_ast_build *build, void *user)
{
  isl_id *id;
  struct polysa_at_domain_data *data = user;
  struct polysa_ast_node_userinfo *node_info;

  node_info = alloc_ast_node_userinfo();
  id = isl_id_alloc(isl_ast_build_get_ctx(build), "", node_info);
  id = isl_id_set_free_user(id, free_ast_node_userinfo);

  if (data->under_pipeline) {
    if (!data->in_pipeline_for) {
      data->in_pipeline_for = 1;
      node_info->is_pipeline = 1;
    }
  } 
  if (data->under_unroll) {
    if (!data->in_unroll_for) {
      data->in_unroll_for = 1;
      node_info->is_unroll = 1;
    }
  }

  return id;
}

static __isl_give isl_ast_node *after_for_module(
  __isl_take isl_ast_node *node, __isl_keep isl_ast_build *build,
  void *user)
{
  isl_id *id;
  struct polysa_ast_node_userinfo *node_info;
  struct polysa_at_domain_data *data = user;

  id = isl_ast_node_get_annotation(node);
  node_info = isl_id_get_user(id);
  if (node_info->is_pipeline) {
    data->in_pipeline_for = 0;
  }
  if (node_info->is_unroll) {
    data->in_unroll_for = 0;
  }

  isl_id_free(id);

  return node;
}

/* Use isl to generate code for both the host and the device
 * from "schedule".
 * The device code is marked by "kernel" mark nodes in the schedule tree,
 * containing a pointer to a polysa_kernel object.
 * The returned AST only contains the AST for the host code.
 * The ASTs for the device code are embedded in polysa_kernel objects
 * attached to the leaf nodes that call "kernel".
 */
__isl_give isl_ast_node *sa_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule)
{
  struct polysa_at_domain_data data;
  isl_ast_build *build;
  isl_ast_node *tree;
  isl_id_list *iterators;
  int depth;

  if (schedule == NULL)
    return NULL;

  data.prog = gen->prog;
  data.kernel = NULL;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  isl_printer_free(p);
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth, 
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark, &data);
  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

/* There are three schedules to handle in this module:
 * - outer loop schdule
 * - inter trans schedule
 * - intra trans schedule
 * We will first generate AST for inter trans function and intra trans function.
 * The AST trees below the inter trans and intra trans marker are stored 
 * seperately.
 * The outer loop AST will print out these two AST trees while handling 
 * the inter trans and intra trans function calls.
 */
isl_stat sa_filter_buffer_io_module_generate_code(struct polysa_gen *gen,
  struct polysa_hw_module *module)
{
  isl_schedule *schedule;
  struct polysa_at_domain_data data;
  isl_ast_build *build;
  isl_ast_node *tree;
  isl_id_list *iterators;
  int depth;
  isl_ctx *ctx;

  /* Generate AST for inter transfer function call. */
  schedule = module->inter_sched;
  ctx = gen->ctx;

  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.filter_buffer = 0;
  data.under_unroll = 0;
  data.under_pipeline = 0;
  data.in_unroll_for = 0;
  data.in_pipeline_for = 0;

//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  // debug 

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_module, &data);
//  build = isl_ast_build_set_before_each_for(build, &before_for_module, &data);
//  build = isl_ast_build_set_after_each_for(build, &after_for_module, &data);

  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);
  isl_ast_node_free(tree);

//  // debug
//  p = isl_printer_print_ast_node(p, module->inter_tree);
//  printf("\n");
//  // debug
 
  /* Generate AST for intra transfer function call. */
  schedule = module->intra_sched;
  ctx = gen->ctx;

  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.filter_buffer = 0;
  data.under_unroll = 0;
  data.under_pipeline = 0;
  data.in_unroll_for = 0;
  data.in_pipeline_for = 0;

//  // debug
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_module, &data);
//  build = isl_ast_build_set_before_each_for(build, &before_for_module, &data);
//  build = isl_ast_build_set_after_each_for(build, &after_for_module, &data);

  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);
  isl_ast_node_free(tree);
 
  /* Generate AST for outer loop function call. */
  schedule = module->outer_sched;
  ctx = gen->ctx;
  
  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.filter_buffer = 0;
  data.under_unroll = 0;
  data.under_pipeline = 0;
  data.in_unroll_for = 0;
  data.in_pipeline_for = 0;

//  // debug
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_module, &data);
//  build = isl_ast_build_set_before_each_for(build, &before_for_module, &data);
//  build = isl_ast_build_set_after_each_for(build, &after_for_module, &data);

  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);
  module->tree = tree;

//  // debug
//  p = isl_printer_free(p);
//  // debug

  return isl_stat_ok;
}

/* Use isl to generate code for the hw module from "schedule".
 * The device code of the hw module is marked by "module" mark nodes in the schedule tree,
 * containing a pointer to a polysa_hw_module object.
 * The returned AST only contains the AST for the host code.
 * The ASTs for the device code are embedded in polysa_hw_module objects
 * attached to the leaf nodes that call "module".
 */
__isl_give isl_ast_node *sa_module_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule)
{
  struct polysa_at_domain_data data;
  isl_ast_build *build;
  isl_ast_node *tree;
  isl_id_list *iterators;

  int depth;

  if (schedule == NULL)
    return NULL;

  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.filter_buffer = 0;
  data.under_unroll = 0;
  data.under_pipeline = 0;
  data.in_unroll_for = 0;
  data.in_pipeline_for = 0;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  p = isl_printer_free(p);
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_module, &data);
//  build = isl_ast_build_set_before_each_for(build, &before_for_module, &data);
//  build = isl_ast_build_set_after_each_for(build, &after_for_module, &data);

  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);

  return tree;
}

__isl_give isl_ast_node *sa_fifo_decl_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule)
{
  struct polysa_at_domain_data data;
  isl_ast_build *build;
  isl_ast_node *tree;
  isl_id_list *iterators;

  int depth;

  if (schedule == NULL)
    return NULL;

  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.top = gen->hw_top_module;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
////  p = isl_printer_free(p);
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_fifo_decl, &data);
  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);

//  // debug
//  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
//  p = isl_printer_print_ast_node(p, tree);
//  printf("\n");
//  p = isl_printer_free(p);
//  // debug

  return tree; 
}

__isl_give isl_ast_node *sa_module_call_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule)
{
  struct polysa_at_domain_data data;
  isl_ast_build *build;
  isl_ast_node *tree;
  isl_id_list *iterators;

  int depth;

  if (schedule == NULL)
    return NULL;

  data.prog = gen->prog;
  data.kernel = NULL;
  data.module = NULL;
  data.top = gen->hw_top_module;

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
////  p = isl_printer_free(p);
//  // debug

  depth = 0;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
        &depth) < 0)
    schedule = isl_schedule_free(schedule);
  build = isl_ast_build_alloc(gen->prog->ctx);
  iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
  build = isl_ast_build_set_iterators(build, iterators);
  build = isl_ast_build_set_at_each_domain(build, &at_domain_module, &data);
  build = isl_ast_build_set_before_each_mark(build, &before_mark_module, &data);
  build = isl_ast_build_set_after_each_mark(build, &after_mark_module_call, &data);
  if (gen->prog->scop->options->debug->dump_final_schedule)
    isl_schedule_dump(schedule);
  tree = isl_ast_build_node_from_schedule(build, schedule);
  isl_ast_build_free(build);

//  // debug
//  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
//  p = isl_printer_print_ast_node(p, tree);
//  printf("\n");
//  p = isl_printer_free(p);
//  // debug

  return tree; 
}
