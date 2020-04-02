#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/vec.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <isl/id_to_id.h>
#include <pet.h>
#include <pet/expr.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "print.h"
#include "schedule.h"
#include "util.h"
#include "polysa_t2s.h"

static struct t2s_array_ref_group *t2s_array_ref_group_free(
  struct t2s_array_ref_group *group
) {
  if (!group)
    return NULL;

  isl_map_free(group->access);
  if (group->n_ref > 1)
    free(group->refs);
  free(group);

  return NULL;
}

static struct t2s_group_data *t2s_group_data_free(struct t2s_group_data *d)
{
  if (!d)
    return NULL;

  isl_union_map_free(d->full_sched);
  free(d);
  
  return NULL;
}

static void ppcg_stmt_free(void *user)
{
	struct ppcg_stmt *stmt = user;

	if (!stmt)
		return;

	isl_id_to_ast_expr_free(stmt->ref2expr);

	free(stmt);
}

static void t2s_stmt_free(void *user)
{
  struct t2s_stmt *stmt = user;

  if (!stmt)
    return;

  free(stmt);
}

static void polysa_t2s_stmt_free(void *user)
{
  struct polysa_t2s_stmt *p_stmt = user;
  
  if (!p_stmt)
    return;

  ppcg_stmt_free(p_stmt->stmt);
  t2s_stmt_free(p_stmt->t_stmt);
  free(p_stmt);
}

/* Mark if the scheudule at each depth is a sequential node or not. */
static isl_bool update_seq_band(__isl_keep isl_schedule_node *node, void *user)
{
  enum isl_schedule_node_type node_type = isl_schedule_node_get_type(node);
  enum isl_schedule_node_type *type_depth = user;
  int total_band_depth = isl_schedule_node_get_schedule_depth(node);
  int total_seq_depth = 0;
  isl_schedule_node *node_tmp = isl_schedule_node_copy(node);
  while (isl_schedule_node_has_parent(node_tmp)) {
    node_tmp = isl_schedule_node_parent(node_tmp);
    if (isl_schedule_node_get_type(node_tmp) == isl_schedule_node_sequence)
      total_seq_depth += 1;
  }
  isl_schedule_node_free(node_tmp);
  int cur_depth = total_band_depth + total_seq_depth;

  if (node_type == isl_schedule_node_band) {
    for (int i = 0; i < isl_schedule_node_band_n_member(node); i++) {
      type_depth[cur_depth + i] = node_type;
    }
  } else if (node_type == isl_schedule_node_sequence) {
    type_depth[cur_depth + 0] = node_type;
  }

  return isl_bool_true;
}

/* Peel off the iterators for scalar dimensions in a vector. */
static __isl_give isl_vec *t2s_peel_off_scalar_dims_vec(__isl_take isl_vec *vec, __isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_union_map *full_sched = isl_schedule_node_get_subtree_schedule_union_map(root);
  isl_set *sched_range = isl_set_from_union_set(isl_union_map_range(full_sched));
  int sched_depth = isl_set_dim(sched_range, isl_dim_set);
  isl_set_free(sched_range);
  isl_schedule_node_free(root);
  isl_ctx *ctx = isl_vec_get_ctx(vec);

  enum isl_schedule_node_type *type_depth = isl_calloc_array(isl_schedule_get_ctx(schedule), 
      enum isl_schedule_node_type, sched_depth);
  for (int i = 0; i < sched_depth; i++) {
    type_depth[i] = -1;
  }

  isl_schedule_foreach_schedule_node_top_down(
      schedule, &update_seq_band, type_depth);

  isl_vec *new_vec = isl_vec_alloc(isl_vec_get_ctx(vec), 0);
  for (int i = 0; i < sched_depth; i++) {
    if (type_depth[i] != isl_schedule_node_sequence) {
      isl_vec *vec_i = isl_vec_alloc(ctx, 1);
      vec_i = isl_vec_set_element_val(vec_i, 0, isl_vec_get_element_val(vec, i));
      new_vec = isl_vec_concat(new_vec, vec_i);
    }
  }

  free(type_depth);
  isl_vec_free(vec);
  return new_vec;
}

/* Peel off the iterators for scalar dimenisions in the iteration domain "set". */
static __isl_give isl_set *t2s_peel_off_scalar_dims(__isl_take isl_set *set, __isl_keep isl_schedule *schedule)
{
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_union_map *full_sched = isl_schedule_node_get_subtree_schedule_union_map(root);
  isl_set *sched_range = isl_set_from_union_set(isl_union_map_range(full_sched));
  int sched_depth = isl_set_dim(sched_range, isl_dim_set);
  isl_set_free(sched_range);
  isl_schedule_node_free(root);

  enum isl_schedule_node_type *type_depth = isl_calloc_array(isl_schedule_get_ctx(schedule), 
      enum isl_schedule_node_type, sched_depth);
  for (int i = 0; i < sched_depth; i++) {
    type_depth[i] = -1;
  }

  isl_schedule_foreach_schedule_node_top_down(
      schedule, &update_seq_band, type_depth);

  int proj_dim = 0;
  for (int i = 0; i < sched_depth; i++) {
    if (type_depth[i] == isl_schedule_node_sequence) {
      set = isl_set_project_out(set, isl_dim_set, i - proj_dim, 1);
      proj_dim++;
    }
  }

  free(type_depth);
  return set;
}

/* Derive the output file name from the input file name.
 * 'input' is the entire path of the input file. The output
 * is the file name plus the additional extension.
 *
 * We will basically replace everything after the last point
 * with '.polysa.c'. This means file.c becomes file.polysa.c
 */
static FILE *get_output_file(const char *input, const char *output)
{
	char name[PATH_MAX];
	const char *ext;
	const char ppcg_marker[] = ".polysa";
	int len;
	FILE *file;

	len = ppcg_extract_base_name(name, input);

	strcpy(name + len, ppcg_marker);
	ext = strrchr(input, '.');
	strcpy(name + len + sizeof(ppcg_marker) - 1, ext ? ext : ".c");

	if (!output)
		output = name;

	file = fopen(output, "w");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for writing\n", output);
		return NULL;
	}

	return file;
}

/* Data used to annotate for nodes in the ast.
 */
struct ast_node_userinfo {
	/* The for node is an openmp parallel for node. */
	int is_openmp;
};

/* Information used while building the ast.
 */
struct ast_build_userinfo {
	/* The current ppcg scop. */
	struct ppcg_scop *scop;

	/* Are we currently in a parallel for loop? */
	int in_parallel_for;
};

/* Check if the current scheduling dimension is parallel.
 *
 * We check for parallelism by verifying that the loop does not carry any
 * dependences.
 * If the live_range_reordering option is set, then this currently
 * includes the order dependences.  In principle, non-zero order dependences
 * could be allowed, but this would require privatization and/or expansion.
 *
 * Parallelism test: if the distance is zero in all outer dimensions, then it
 * has to be zero in the current dimension as well.
 * Implementation: first, translate dependences into time space, then force
 * outer dimensions to be equal.  If the distance is zero in the current
 * dimension, then the loop is parallel.
 * The distance is zero in the current dimension if it is a subset of a map
 * with equal values for the current dimension.
 */
static int ast_schedule_dim_is_parallel(__isl_keep isl_ast_build *build,
	struct ppcg_scop *scop)
{
	isl_union_map *schedule, *deps;
	isl_map *schedule_deps, *test;
	isl_space *schedule_space;
	unsigned i, dimension, is_parallel;

	schedule = isl_ast_build_get_schedule(build);
	schedule_space = isl_ast_build_get_schedule_space(build);

	dimension = isl_space_dim(schedule_space, isl_dim_out) - 1;

	deps = isl_union_map_copy(scop->dep_flow);
	deps = isl_union_map_union(deps, isl_union_map_copy(scop->dep_false));
	if (scop->options->live_range_reordering) {
		isl_union_map *order = isl_union_map_copy(scop->dep_order);
		deps = isl_union_map_union(deps, order);
	}
	deps = isl_union_map_apply_range(deps, isl_union_map_copy(schedule));
	deps = isl_union_map_apply_domain(deps, schedule);

	if (isl_union_map_is_empty(deps)) {
		isl_union_map_free(deps);
		isl_space_free(schedule_space);
		return 1;
	}

	schedule_deps = isl_map_from_union_map(deps);

	for (i = 0; i < dimension; i++)
		schedule_deps = isl_map_equate(schedule_deps, isl_dim_out, i,
					       isl_dim_in, i);

	test = isl_map_universe(isl_map_get_space(schedule_deps));
	test = isl_map_equate(test, isl_dim_out, dimension, isl_dim_in,
			      dimension);
	is_parallel = isl_map_is_subset(schedule_deps, test);

	isl_space_free(schedule_space);
	isl_map_free(test);
	isl_map_free(schedule_deps);

	return is_parallel;
}

/* Mark a for node openmp parallel, if it is the outermost parallel for node.
 */
static void mark_openmp_parallel(__isl_keep isl_ast_build *build,
	struct ast_build_userinfo *build_info,
	struct ast_node_userinfo *node_info)
{
	if (build_info->in_parallel_for)
		return;

	if (ast_schedule_dim_is_parallel(build, build_info->scop)) {
		build_info->in_parallel_for = 1;
		node_info->is_openmp = 1;
	}
}

/* Allocate an ast_node_info structure and initialize it with default values.
 */
static struct ast_node_userinfo *allocate_ast_node_userinfo()
{
	struct ast_node_userinfo *node_info;
	node_info = (struct ast_node_userinfo *)
		malloc(sizeof(struct ast_node_userinfo));
	node_info->is_openmp = 0;
	return node_info;
}

/* Free an ast_node_info structure.
 */
static void free_ast_node_userinfo(void *ptr)
{
	struct ast_node_userinfo *info;
	info = (struct ast_node_userinfo *) ptr;
	free(info);
}

/* This method is executed before the construction of a for node. It creates
 * an isl_id that is used to annotate the subsequently generated ast for nodes.
 *
 * In this function we also run the following analyses:
 *
 * 	- Detection of openmp parallel loops
 */
static __isl_give isl_id *ast_build_before_for(
	__isl_keep isl_ast_build *build, void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *node_info;

	build_info = (struct ast_build_userinfo *) user;
	node_info = allocate_ast_node_userinfo();
	id = isl_id_alloc(isl_ast_build_get_ctx(build), "", node_info);
	id = isl_id_set_free_user(id, free_ast_node_userinfo);

	mark_openmp_parallel(build, build_info, node_info);

	return id;
}

/* This method is executed after the construction of a for node.
 *
 * It performs the following actions:
 *
 * 	- Reset the 'in_parallel_for' flag, as soon as we leave a for node,
 * 	  that is marked as openmp parallel.
 *
 */
static __isl_give isl_ast_node *ast_build_after_for(
	__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build,
	void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *info;

	id = isl_ast_node_get_annotation(node);
	info = isl_id_get_user(id);

	if (info && info->is_openmp) {
		build_info = (struct ast_build_userinfo *) user;
		build_info->in_parallel_for = 0;
	}

	isl_id_free(id);

	return node;
}

/* Find the element in scop->stmts that has the given "id".
 */
static struct pet_stmt *find_pet_stmt(struct ppcg_scop *scop, __isl_keep isl_id *id)
{
	int i;

	for (i = 0; i < scop->pet->n_stmt; ++i) {
		struct pet_stmt *stmt = scop->pet->stmts[i];
		isl_id *id_i;

		id_i = isl_set_get_tuple_id(stmt->domain);
		isl_id_free(id_i);

		if (id_i == id)
			return stmt;
	}

	isl_die(isl_id_get_ctx(id), isl_error_internal,
		"statement not found", return NULL);
}

/* Print a user statement in the generated AST.
 * The ppcg_stmt has been attached to the node in at_each_domain.
 */
static __isl_give isl_printer *print_user(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;

	id = isl_ast_node_get_annotation(node);
  stmt = isl_id_get_user(id);
	isl_id_free(id);

  p = pet_stmt_print_body(stmt->stmt, p, stmt->ref2expr);

	isl_ast_print_options_free(print_options);

	return p;
}


/* Print a for loop node as an openmp parallel loop.
 *
 * To print an openmp parallel loop we print a normal for loop, but add
 * "#pragma openmp parallel for" in front.
 *
 * Variables that are declared within the body of this for loop are
 * automatically openmp 'private'. Iterators declared outside of the
 * for loop are automatically openmp 'shared'. As ppcg declares all iterators
 * at the position where they are assigned, there is no need to explicitly mark
 * variables. Their automatically assigned type is already correct.
 *
 * This function only generates valid OpenMP code, if the ast was generated
 * with the 'atomic-bounds' option enabled.
 *
 */
static __isl_give isl_printer *print_for_with_openmp(
	__isl_keep isl_ast_node *node, __isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "#pragma omp parallel for");
	p = isl_printer_end_line(p);

	p = isl_ast_node_for_print(node, p, print_options);

	return p;
}

/* Print a for node.
 *
 * Depending on how the node is annotated, we either print a normal
 * for node or an openmp parallel for node.
 */
static __isl_give isl_printer *print_for(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
	int openmp;

	openmp = 0;
	id = isl_ast_node_get_annotation(node);

	if (id) {
		struct ast_node_userinfo *info;

		info = (struct ast_node_userinfo *) isl_id_get_user(id);
		if (info && info->is_openmp)
			openmp = 1;
	}

	if (openmp)
		p = print_for_with_openmp(node, p, print_options);
	else
		p = isl_ast_node_for_print(node, p, print_options);

	isl_id_free(id);

	return p;
}

/* Index transformation callback for pet_stmt_build_ast_exprs.
 *
 * "index" expresses the array indices in terms of statement iterators
 * "iterator_map" expresses the statement iterators in terms of
 * AST loop iterators.
 *
 * The result expresses the array indices in terms of
 * AST loop iterators.
 */
static __isl_give isl_multi_pw_aff *pullback_index(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *id, void *user)
{
	isl_pw_multi_aff *iterator_map = user;

	iterator_map = isl_pw_multi_aff_copy(iterator_map);
	return isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);
}

/* Transform the accesses in the statement associated to the domain
 * called by "node" to refer to the AST loop iterators, construct
 * corresponding AST expressions using "build",
 * collect them in a ppcg_stmt and annotate the node with the ppcg_stmt.
 */
static __isl_give isl_ast_node *at_each_domain(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct ppcg_scop *scop = user;
	isl_ast_expr *expr, *arg;
	isl_ctx *ctx;
	isl_id *id;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	struct ppcg_stmt *stmt;

  ctx = isl_ast_node_get_ctx(node);
	stmt = isl_calloc_type(ctx, struct ppcg_stmt);   
	if (!stmt)
		goto error;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
  isl_ast_expr_free(expr);
	id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(arg);
	stmt->stmt = find_pet_stmt(scop, id);
  isl_id_free(id);   
	if (!stmt->stmt)
		goto error;

	map = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	map = isl_map_reverse(map);
	iterator_map = isl_pw_multi_aff_from_map(map);
	stmt->ref2expr = pet_stmt_build_ast_exprs(stmt->stmt, build,
				    &pullback_index, iterator_map, NULL, NULL);
	isl_pw_multi_aff_free(iterator_map);

  id = isl_id_alloc(isl_ast_node_get_ctx(node), NULL, stmt);
	id = isl_id_set_free_user(id, &ppcg_stmt_free);
	return isl_ast_node_set_annotation(node, id);
error:
	ppcg_stmt_free(stmt);
	return isl_ast_node_free(node);
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

/* This function is called for each node in a CPU AST.
 * In case of a user node, print the macro definitions required
 * for printing the AST expressions in the annotation, if any.
 * For other nodes, return true such that descendants are also
 * visited.
 *
 * In particular, print the macro definitions needed for the substitutions
 * of the original user statements.
 */
static isl_bool at_node(__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	if (!stmt)
		return isl_bool_error;

	*p = ppcg_print_body_macros(*p, stmt->ref2expr);
	if (!*p)
		return isl_bool_error;

	return isl_bool_false;
}

/* Print the required macros for the CPU AST "node" to "p",
 * including those needed for the user statements inside the AST.
 */
static __isl_give isl_printer *cpu_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_descendant_top_down(node, &at_node, &p) < 0)
		return isl_printer_free(p);
	p = ppcg_print_macros(p, node);
	return p;
}

static isl_bool debug_ast_node(__isl_keep isl_ast_node *node, void *user)
{
  enum isl_ast_node_type type = isl_ast_node_get_type(node);
  switch (type) {
    case isl_ast_node_user:
      printf("user node found.\n");
      break;
  }

  return isl_bool_true;
}

/* Code generate the scop 'scop' using "schedule"
 * and print the corresponding C code to 'p'.
 */
static __isl_give isl_printer *print_scop(struct ppcg_scop *scop,
	__isl_take isl_schedule *schedule, __isl_take isl_printer *p,
	struct ppcg_options *options)
{
	isl_ctx *ctx = isl_printer_get_ctx(p);
	isl_ast_build *build;
	isl_ast_print_options *print_options;
	isl_ast_node *tree;
	isl_id_list *iterators;
	struct ast_build_userinfo build_info;
	int depth;

	depth = 0;
	if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
						&depth) < 0)
		goto error;

	build = isl_ast_build_alloc(ctx);
	iterators = ppcg_scop_generate_names(scop, depth, "c");
	build = isl_ast_build_set_iterators(build, iterators);
	build = isl_ast_build_set_at_each_domain(build, &at_each_domain, scop); 

	if (options->openmp) {
		build_info.scop = scop;
		build_info.in_parallel_for = 0;

		build = isl_ast_build_set_before_each_for(build,
							&ast_build_before_for,
							&build_info);
		build = isl_ast_build_set_after_each_for(build,
							&ast_build_after_for,
							&build_info);
	}

	tree = isl_ast_build_node_from_schedule(build, schedule);
	isl_ast_build_free(build);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_user, NULL);

	print_options = isl_ast_print_options_set_print_for(print_options,
							&print_for, NULL);

	p = cpu_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	isl_ast_node_free(tree);

	return p;
error:
	isl_schedule_free(schedule);
	isl_printer_free(p);
	return NULL;
}

/* Tile the band node "node" with tile sizes "sizes" and
 * mark all members of the resulting tile node as "atomic".
 */
static __isl_give isl_schedule_node *tile(__isl_take isl_schedule_node *node,
	__isl_take isl_multi_val *sizes)
{
	node = isl_schedule_node_band_tile(node, sizes);
	node = ppcg_set_schedule_node_type(node, isl_ast_loop_atomic);

	return node;
}

/* Tile "node", if it is a band node with at least 2 members.
 * The tile sizes are set from the "tile_size" option.
 */
static __isl_give isl_schedule_node *t2s_tile_band(
	__isl_take isl_schedule_node *node, void *user)
{
	struct ppcg_scop *scop = user;
	int n;
	isl_space *space;
	isl_multi_val *sizes;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;

	n = isl_schedule_node_band_n_member(node);
	if (n <= 1)
		return node;

	space = isl_schedule_node_band_get_space(node);
	sizes = ppcg_multi_val_from_int(space, scop->options->tile_size);

	return tile(node, sizes);
}

/* Construct schedule constraints from the dependences in ps
 * for the purpose of computing a schedule for a CPU.
 *
 * The proximity constraints are set to the flow dependences.
 *
 * If live-range reordering is allowed then the conditional validity
 * constraints are set to the order dependences with the flow dependences
 * as condition.  That is, a live-range (flow dependence) will be either
 * local to an iteration of a band or all adjacent order dependences
 * will be respected by the band.
 * The validity constraints are set to the union of the flow dependences
 * and the forced dependences, while the coincidence constraints
 * are set to the union of the flow dependences, the forced dependences and
 * the order dependences.
 *
 * If live-range reordering is not allowed, then both the validity
 * and the coincidence constraints are set to the union of the flow
 * dependences and the false dependences.
 *
 * Note that the coincidence constraints are only set when the "openmp"
 * options is set.  Even though the way openmp pragmas are introduced
 * does not rely on the coincident property of the schedule band members,
 * the coincidence constraints do affect the way the schedule is constructed,
 * such that more schedule dimensions should be detected as parallel
 * by ast_schedule_dim_is_parallel.
 * Since the order dependences are also taken into account by
 * ast_schedule_dim_is_parallel, they are also added to
 * the coincidence constraints.  If the openmp handling learns
 * how to privatize some memory, then the corresponding order
 * dependences can be removed from the coincidence constraints.
 */
static __isl_give isl_schedule_constraints *construct_cpu_schedule_constraints(
	struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_union_map *validity, *coincidence;

	sc = isl_schedule_constraints_on_domain(isl_union_set_copy(ps->domain));
	if (ps->options->live_range_reordering) {
		sc = isl_schedule_constraints_set_conditional_validity(sc,
				isl_union_map_copy(ps->tagged_dep_flow),
				isl_union_map_copy(ps->tagged_dep_order));
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_forced));
//		if (ps->options->openmp) {
			coincidence = isl_union_map_copy(validity);
			coincidence = isl_union_map_union(coincidence,
					isl_union_map_copy(ps->dep_order));
//		}
    /* Add the RAR dependences into the validity constraints for
     * systolic array generation.
     */
    if (ps->options->polysa) {
      validity = isl_union_map_union(validity,
          isl_union_map_copy(ps->dep_rar));
    }   
	} else {
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_false));
//		if (ps->options->openmp)
			coincidence = isl_union_map_copy(validity);
    /* Add the RAR dependences into the validity constraints for 
     * systolic array generation.
     */
    if (ps->options->polysa) {
      validity = isl_union_map_union(validity,
          isl_union_map_copy(ps->dep_rar));
    }   
	}
//	if (ps->options->openmp)
		sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_proximity(sc,
					isl_union_map_copy(ps->dep_flow));

	return sc;
}

/* Compute a schedule for the scop "ps".
 *
 * First derive the appropriate schedule constraints from the dependences
 * in "ps" and then compute a schedule from those schedule constraints,
 * possibly grouping statement instances based on the input schedule.
 */
static __isl_give isl_schedule *compute_cpu_schedule(struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	sc = construct_cpu_schedule_constraints(ps);

	schedule = ppcg_compute_schedule(sc, ps->schedule, ps->options);

	return schedule;
}

/* Compute a new schedule to the scop "ps" if the reschedule option is set.
 * Otherwise, return a copy of the original schedule.
 */
static __isl_give isl_schedule *optionally_compute_schedule(void *user)
{
	struct ppcg_scop *ps = user;

	if (!ps)
		return NULL;
	if (!ps->options->reschedule)
		return isl_schedule_copy(ps->schedule);
	return compute_cpu_schedule(ps);
}

/* Compute a schedule based on the dependences in "ps" and
 * tile it if requested by the user.
 */
static __isl_give isl_schedule *t2s_get_schedule(struct ppcg_scop *ps,
	struct ppcg_options *options)
{
	isl_ctx *ctx;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	ctx = isl_union_set_get_ctx(ps->domain);
	schedule = ppcg_get_schedule(ctx, options,
				    &optionally_compute_schedule, ps);
	if (ps->options->tile)
		schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
							&t2s_tile_band, ps);

	return schedule;
}

/* Generate CPU code for the scop "ps" using "schedule" and
 * print the corresponding C code to "p", including variable declarations.
 */
static __isl_give isl_printer *print_cpu_with_schedule(
	__isl_take isl_printer *p, struct ppcg_scop *ps,
	__isl_take isl_schedule *schedule, struct ppcg_options *options)
{
	int hidden;
	isl_set *context;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "/* PPCG generated CPU code */");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	p = ppcg_set_macro_names(p);
	p = ppcg_print_exposed_declarations(p, ps);
	hidden = ppcg_scop_any_hidden_declarations(ps);
	if (hidden) {
		p = ppcg_start_block(p);
		p = ppcg_print_hidden_declarations(p, ps);
	}

	context = isl_set_copy(ps->context);
	context = isl_set_from_params(context);
	schedule = isl_schedule_insert_context(schedule, context);
	if (options->debug->dump_final_schedule)
		isl_schedule_dump(schedule);
	p = print_scop(ps, schedule, p, options);
	if (hidden)
		p = ppcg_end_block(p);

	return p;
}

//static __isl_give isl_schedule_node *aggregate_stmt_domain(__isl_take isl_schedule_node *node, void *user)
//{
//  isl_union_set *domain;
//  isl_union_map *schedule;
//  isl_set *stmt_domain;
//  isl_set **anchor_domain = (isl_set **)(user);
//
//  if (!node)
//    return NULL;
//
//  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
//    return node;
//
//  domain = isl_schedule_node_get_domain(node);
//  schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
//  schedule = isl_union_map_intersect_domain(schedule, domain);
//  stmt_domain = isl_set_from_union_set(isl_union_map_range(schedule));
//  if (*anchor_domain == NULL)
//    *anchor_domain = isl_set_copy(stmt_domain);
//  else
//    *anchor_domain = isl_set_union(*anchor_domain, isl_set_copy(stmt_domain));
//
//  isl_set_free(stmt_domain);
//
//  return node;
//}

/* Extract the (simplified) iteration domain of each user statement. */
static isl_stat extract_each_stmt_domain(__isl_take isl_set *set, void *user)
{
  struct t2s_data *data = user;
  isl_union_set *sched_domain = isl_union_set_apply(isl_union_set_from_set(isl_set_copy(set)), isl_union_map_copy(data->sched));
  isl_set *stmt_domain_i = isl_set_from_union_set(sched_domain);
  isl_set *stmt_sim_domain_i = isl_set_gist(isl_set_copy(stmt_domain_i), 
      isl_set_copy(data->anchor_domain));

  /* Set the name of space. */
  isl_space *space = isl_set_get_space(set);
  const char *stmt_name = isl_space_get_tuple_name(space, isl_dim_set);
  stmt_domain_i = isl_set_set_tuple_name(stmt_domain_i, stmt_name);  
  stmt_sim_domain_i = isl_set_set_tuple_name(stmt_sim_domain_i, stmt_name);
  
  isl_set_free(set);
  isl_space_free(space);

  if (data->stmt_domain == NULL)
    data->stmt_domain = isl_union_set_from_set(stmt_domain_i);
  else
    data->stmt_domain = isl_union_set_union(data->stmt_domain, isl_union_set_from_set(stmt_domain_i));

  if (data->stmt_sim_domain == NULL)
    data->stmt_sim_domain = isl_union_set_from_set(stmt_sim_domain_i);
  else
    data->stmt_sim_domain = isl_union_set_union(data->stmt_sim_domain,
        isl_union_set_from_set(stmt_sim_domain_i));

  return isl_stat_ok;
}

/* Extract the (simplified) iteration domain of all the user statemets. */
static isl_stat extract_stmt_domain(__isl_keep isl_schedule *schedule, struct t2s_data *data)
{
  isl_union_set *domain = isl_schedule_get_domain(schedule);
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(root);
  data->sched = sched;
  isl_schedule_node_free(root);

  /* Assign the scheduling space the same name as the statement. */
  isl_union_set_foreach_set(domain, &extract_each_stmt_domain, data);

  isl_union_set_free(domain);
  data->sched = isl_union_map_free(data->sched);
}

/* Duplicate the polysa_dep. */
__isl_give struct polysa_dep *polysa_dep_copy(__isl_keep struct polysa_dep *dep)
{
  struct polysa_dep *new_dep = (struct polysa_dep *)malloc(sizeof(struct polysa_dep));
  new_dep->src = isl_id_copy(dep->src);
  new_dep->dest = isl_id_copy(dep->dest);
  new_dep->disvec = isl_vec_copy(dep->disvec);
  new_dep->isl_dep = isl_basic_map_copy(dep->isl_dep);
  new_dep->type = dep->type;
  new_dep->src_sched_domain = isl_set_copy(dep->src_sched_domain);
  new_dep->dest_sched_domain = isl_set_copy(dep->dest_sched_domain);

  return new_dep;
}

/* This function extracts the raw and rar deps that have the dest access associated 
 * with the current access.
 */
static int t2s_update_dep(__isl_keep pet_expr *expr, void *user)
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data = data->stmt_data;

  isl_id *id;
  id = isl_id_copy(expr->acc.ref_id);
 
  int n;
  for (n = 0; n < data->ndeps; n++) {
    struct polysa_dep *dep_i = data->deps[n];
    if (dep_i->dest == id && dep_i->type == POLYSA_DEP_RAW)
      break;
  }

  if (n != data->ndeps) {
    stmt_data->stmt_deps = (struct polysa_dep ***)realloc(stmt_data->stmt_deps, (stmt_data->n_acc_group + 1) * sizeof(struct polysa_dep **));   
    stmt_data->stmt_deps[stmt_data->n_acc_group] = NULL;
    stmt_data->n_dep_per_acc_group = (int *)realloc(stmt_data->n_dep_per_acc_group, (stmt_data->n_acc_group + 1) * sizeof(int));
    stmt_data->n_dep_per_acc_group[stmt_data->n_acc_group] = 0;
    for (int i = 0; i < data->ndeps; i++) { 
      struct polysa_dep *dep_i = data->deps[i];
      if (dep_i->dest == id && dep_i->type == POLYSA_DEP_RAW) {
        stmt_data->stmt_deps[stmt_data->n_acc_group] = (struct polysa_dep **)realloc(stmt_data->stmt_deps[stmt_data->n_acc_group], 
            (stmt_data->n_dep_per_acc_group[stmt_data->n_acc_group] + 1) * sizeof(struct polysa_dep *));
        stmt_data->stmt_deps[stmt_data->n_acc_group][stmt_data->n_dep_per_acc_group[stmt_data->n_acc_group]] = polysa_dep_copy(dep_i);
        stmt_data->n_dep_per_acc_group[stmt_data->n_acc_group] += 1;
      }
    }
    
    stmt_data->n_acc_group += 1;
  }

  isl_id_free(id);

  return 0;
}

/* Generate the t2s function for each array access. */
static int t2s_update_access(__isl_keep pet_expr *expr, void *user)
{
  struct t2s_data *data = user;  
  struct t2s_stmt_data *stmt_data = data->stmt_data;
  isl_id *id;
  isl_multi_pw_aff *index;
  isl_space *index_space;
  isl_id *array_id;
  isl_ctx *ctx;
  isl_ast_expr *ast_expr;

  id = isl_id_copy(expr->acc.ref_id);
  index = isl_multi_pw_aff_copy(expr->acc.index);
  ctx = isl_id_get_ctx(id);

  isl_multi_pw_aff_free(index);

  /* If the access is associated with RAR, then generate access as
   * A(c0, c1, c2).
   * If the access is associated with RAW, then generate access as
   * A(c0, c1 - 1, c2).
   * Otherwise, generate A(c0, c1, c2).
   */
  isl_id *func = isl_id_to_id_get(data->ref2func, isl_id_copy(id));
  isl_ast_expr *func_expr = isl_ast_expr_from_id(func);
  isl_ast_expr_list *args = isl_ast_expr_list_alloc(ctx, 0);

  int i;
  for (i = 0; i < stmt_data->n_acc_group; i++) {
    struct polysa_dep *dep_i = stmt_data->dep_stmt_pair[i];
    if (dep_i->dest == id && dep_i->type == POLYSA_DEP_RAW) {
      for (int j = 0; j < data->iter_num; j++) {
        char iter_name[100];
        isl_val *ele = isl_vec_get_element_val(dep_i->disvec, j);
        if (isl_val_is_zero(ele)) {
          sprintf(iter_name, "c%d", j);
        } else {
          sprintf(iter_name, "c%d - %ld", j, isl_val_get_num_si(ele));
        }
        isl_id *arg = isl_id_alloc(ctx, iter_name, NULL);
        isl_ast_expr *arg_expr = isl_ast_expr_from_id(arg);
        args = isl_ast_expr_list_add(args, arg_expr);
        isl_val_free(ele);
      }
      /* Update the func_expr as the src func_expr. */
      func = isl_id_to_id_get(data->ref2func, isl_id_copy(dep_i->src));
      isl_ast_expr_free(func_expr);
      func_expr = isl_ast_expr_from_id(func);

      ast_expr = isl_ast_expr_access(func_expr, args);
      break;
    }
  }
  if (i == stmt_data->n_acc_group) {
    for (int j = 0; j < data->iter_num; j++) {
      char iter_name[100];
      sprintf(iter_name, "c%d", j);
      isl_id *arg = isl_id_alloc(ctx, iter_name, NULL);
      isl_ast_expr *arg_expr = isl_ast_expr_from_id(arg);
      args = isl_ast_expr_list_add(args, arg_expr);
    }
    ast_expr = isl_ast_expr_access(func_expr, args);
  }

  stmt_data->stmts[stmt_data->stmt_num - 1]->ref2expr = 
    isl_id_to_ast_expr_set(stmt_data->stmts[stmt_data->stmt_num - 1]->ref2expr, id, ast_expr);

	return 0;
}

/* Generate a T2S statement for each unique dependence pair. */
isl_bool gen_t2s_stmt(__isl_take struct polysa_dep **dep_stmt_pair, struct ppcg_stmt *stmt, struct t2s_data *data)
{
  struct t2s_stmt_data *stmt_data = data->stmt_data;
  isl_set *union_domain = isl_set_copy(data->stmt_data->stmt_anchor_domain); 
  if (data->stmt_data->n_acc_group > 0) {
    for (int i = 0; i < data->stmt_data->n_acc_group; i++) {
      struct polysa_dep *dep_i = dep_stmt_pair[i];
      isl_set *dest_sched_domain = isl_set_copy(dep_i->dest_sched_domain);
      dest_sched_domain = isl_set_set_tuple_id(dest_sched_domain, isl_set_get_tuple_id(union_domain));
      union_domain = isl_set_intersect(union_domain, dest_sched_domain);
    }

    if (isl_set_is_empty(union_domain)) {
      free(dep_stmt_pair);
      isl_set_free(union_domain);
      return isl_bool_false;
    }
  }

  /* Simplify the domain. */
  isl_set *anchor_domain = isl_set_copy(data->anchor_domain);
  isl_space *space = isl_set_get_space(union_domain);
  anchor_domain = isl_set_set_tuple_name(anchor_domain, isl_space_get_tuple_name(space, isl_dim_set));
  union_domain = isl_set_gist(union_domain, anchor_domain);
  isl_space_free(space);

  /* Peel off the scalar dimensions. */
  union_domain = t2s_peel_off_scalar_dims(union_domain, data->schedule);

  data->stmt_data->stmt_num += 1;
  data->stmt_data->stmts = (struct ppcg_stmt **)realloc(data->stmt_data->stmts, data->stmt_data->stmt_num * sizeof(struct ppcg_stmt *));
  data->stmt_data->stmts[data->stmt_data->stmt_num - 1] = (struct ppcg_stmt *)malloc(sizeof(struct ppcg_stmt));
  data->stmt_data->stmt_domain = (isl_set **)realloc(data->stmt_data->stmt_domain, data->stmt_data->stmt_num * sizeof(isl_set *));
  data->stmt_data->stmt_domain[data->stmt_data->stmt_num - 1] = union_domain;

  stmt_data->stmts[stmt_data->stmt_num - 1]->stmt = stmt->stmt;
  stmt_data->stmts[stmt_data->stmt_num - 1]->ref2expr = isl_id_to_ast_expr_alloc(data->ctx, 0);

  /* Produce the ref2expr for each access. */
  stmt_data->dep_stmt_pair = dep_stmt_pair;
  pet_tree_foreach_access_expr(stmt->stmt->body, &t2s_update_access, data);

  free(dep_stmt_pair);
  stmt_data->dep_stmt_pair = NULL;

  return isl_bool_true;
}

/* This function builds the ref2expr for each array reference in the user statement.
 * The LHS access A[][] is replaced by A(c0, c1, c2).
 * The RHS acesss B[][] is replaced by the flow dep B(c0 - 1, c1, c2).
 * If the RHS access is associated with multiple flow deps, we will need to the split the statement into multiple T2S statement. 
 * At present, we don't support stmts with more than one RHS access that are associated with flow deps.
 *
 * First determine how many unique ref2expr pairs are to be generated. 
 * Then build the ref2expr for each pair.
 */
static isl_stat extract_t2s_stmt_access(__isl_take struct ppcg_stmt *stmt, struct t2s_data *data)
{
  struct t2s_stmt_data *stmt_data = data->stmt_data;
  /* Determine all the unique ref2expr pairs to be generated. */
  pet_tree_foreach_access_expr(stmt->stmt->body, &t2s_update_dep, data);
  int stmt_num = 1;
  for (int i = 0; i < stmt_data->n_acc_group; i++) {
    stmt_num *= stmt_data->n_dep_per_acc_group[i];
  }
  struct polysa_dep ***dep_stmt_pairs = NULL;
  dep_stmt_pairs = (struct polysa_dep ***)malloc(stmt_num * sizeof(struct polysa_dep **));
  for (int i = 0; i < stmt_num; i++) {
    dep_stmt_pairs[i] = (struct polysa_dep **)malloc(stmt_data->n_acc_group * sizeof(struct polysa_dep *));      
  }
  int prev_repeat = 1;
  int post_repeat = stmt_num;
  int acc_group_i = 0;
  while(acc_group_i < stmt_data->n_acc_group) {
    int cur_n_dep_acc_group = stmt_data->n_dep_per_acc_group[acc_group_i];
    post_repeat /= cur_n_dep_acc_group;
    int id = 0;
    for (int i = 0; i < prev_repeat; i++)
      for (int j = 0; j < cur_n_dep_acc_group; j++)
        for (int k = 0; k < post_repeat; k++) {          
          dep_stmt_pairs[id][acc_group_i] = stmt_data->stmt_deps[acc_group_i][j];
          id++;
        }
    acc_group_i++;
    prev_repeat *= cur_n_dep_acc_group; 
  }

  /* Gnerate a seperate ppcg_stmt for each dep pair. */
  stmt_data->stmt_num = 0;
  for (int i = 0; i < stmt_num; i++) {
    isl_bool success = gen_t2s_stmt(dep_stmt_pairs[i], stmt, data);
    if (!success) {
      for (int j = i; j < stmt_num - 1; j++) {
        dep_stmt_pairs[j] = dep_stmt_pairs[j + 1];
      }
      stmt_num -= 1;
      i--;
    }
  }

  ppcg_stmt_free(stmt);
  free(dep_stmt_pairs);

  return isl_stat_ok; 
}

//static char *concat(const char *s1, const char *s2) 
//{
//  char *result = malloc(strlen(s1) + strlen(s2) + 1);
//  strcpy(result, s1);
//  strcat(result, s2);
//  return result;
//}

/* Print an "set" in T2S format. */
static char *isl_set_to_t2s_format(__isl_keep isl_set *set)
{
  char *t2s_cst = NULL;
  int n_bset = isl_set_n_basic_set(set);
  int bset_id = 0;
  int multi_bset = n_bset > 1;
  isl_basic_set_list *bset_list = isl_set_get_basic_set_list(set);
  isl_printer *p = isl_printer_to_str(isl_set_get_ctx(set));

  while (bset_id < n_bset) {
    if (bset_id > 0) {
      p = isl_printer_print_str(p, " || ");      
    }
    if (multi_bset) {
      p = isl_printer_print_str(p, "(");    
    }
    /* Print the content of each basic map. */
    isl_basic_set *bset = isl_basic_set_list_get_basic_set(bset_list, bset_id);
    
    isl_constraint_list *cst_list = isl_basic_set_get_constraint_list(bset);
    int cst_id = 0;
    int n_cst = isl_basic_set_n_constraint(bset);
    int multi_cst = n_cst > 1;
    while (cst_id < n_cst) {
      if (cst_id > 0) {
        p = isl_printer_print_str(p, " && ");
      }
      if (multi_cst) {
        p = isl_printer_print_str(p, "(");
      }

      isl_constraint *cst_i = isl_constraint_list_get_constraint(cst_list, cst_id);

      /* TODO: consider isl_dim_div later. */
      int is_first = 1;
      for (int i = 0; i < isl_constraint_dim(cst_i, isl_dim_set); i++) {        
        isl_val *val = isl_constraint_get_coefficient_val(cst_i, isl_dim_set, i);
        const char *name = isl_constraint_get_dim_name(cst_i, isl_dim_set, i);
        if (!isl_val_is_zero(val)) {
          if (!is_first) {
            p = isl_printer_print_str(p, " + ");
            is_first = 0;
          }
          if (!isl_val_is_one(val)) {
            p = isl_printer_print_val(p, val);
            p = isl_printer_print_str(p, " * ");
          }
          p = isl_printer_print_str(p, name);
          if (is_first)
            is_first = 0;
        }
        isl_val_free(val);
      }
      for (int i = 0; i < isl_constraint_dim(cst_i, isl_dim_param); i++) {
        isl_val *val = isl_constraint_get_coefficient_val(cst_i, isl_dim_param, i);
        const char *name = isl_constraint_get_dim_name(cst_i, isl_dim_param, i);
        if (!isl_val_is_zero(val)) {
          if (!is_first) {
            p = isl_printer_print_str(p, " + ");            
            is_first = 0;
          }
          if (!isl_val_is_one(val)) {
            p = isl_printer_print_val(p, val);
            p = isl_printer_print_str(p, " * ");
          }
          p = isl_printer_print_str(p, name);
          if (is_first)
            is_first = 0;
        }
        isl_val_free(val);
      }
      isl_val *cst_val = isl_constraint_get_constant_val(cst_i);
      if (!isl_val_is_zero(cst_val)) {
        p = isl_printer_print_str(p, " + ");
        p = isl_printer_print_val(p, cst_val);
      }
      isl_val_free(cst_val);
      if (isl_constraint_is_equality(cst_i)) {
        p = isl_printer_print_str(p, " == 0");
      } else {
        p = isl_printer_print_str(p, " >= 0");
      }
      
      isl_constraint_free(cst_i);

      if (multi_cst) {
        p = isl_printer_print_str(p, ")");
      }
      cst_id++;
    }
    isl_constraint_list_free(cst_list);

    if (multi_bset) {
      p = isl_printer_print_str(p, ")");    
    }

    isl_basic_set_free(bset);
    bset_id++;
  }

  isl_basic_set_list_free(bset_list);
  t2s_cst = isl_printer_get_str(p);
  isl_printer_free(p);

  return t2s_cst;
}

/* This function takes in the C statement "c_text" like
 * C[i][j] = 0
 * and the iteration domain "domain", 
 * prints out the T2S statement like
 * C(i,j) = select(i == 0 && j == 0, C(i,j-1), C(i,j))
 */
static __isl_give char *c_to_t2s_stmt(__isl_take char *c_text, __isl_take isl_set *domain, int iter_num) {
  char ch;
  int loc = 0;
  int insert_select;
  char *iter_domain = NULL;
  char *t2s_text = NULL;
  isl_printer *p = isl_printer_to_str(isl_set_get_ctx(domain));
  char *LHS_func = NULL;
  int at_LHS = 1;
  isl_ctx *ctx = isl_set_get_ctx(domain);
  isl_printer *p_LHS = isl_printer_to_str(ctx);

  /* Generate the iteration domain constructs in T2S format. */
  if (!isl_set_is_empty(domain)) {
    /* Set up the iterators. */
    for (int i = 0; i < isl_set_dim(domain, isl_dim_set); i++) {
      char iter_name[100];
      sprintf(iter_name, "c%d", i);
      isl_id *id_i = isl_id_alloc(ctx, iter_name, NULL);      
      domain = isl_set_set_dim_id(domain, isl_dim_set, i, id_i);
    }
    iter_domain = isl_set_to_t2s_format(domain); 
  }
 
  while ((ch = c_text[loc]) != '\0') { 
    if (ch == '=') {
      while((ch = c_text[++loc]) == ' ') {
        ;
      }
      p = isl_printer_print_str(p, "= ");
      if (iter_domain) {
        p = isl_printer_print_str(p, "select(");
        p = isl_printer_print_str(p, iter_domain);
        p = isl_printer_print_str(p, ", ");
      }
    } else if (ch == ';') {    
      if (iter_domain) {
//        p = isl_printer_print_str(p, ", ");
//        p = isl_printer_print_str(p, LHS_func);
        p = isl_printer_print_str(p, ")");
      }
    } else if (ch == '[') {
      isl_printer *p_func = isl_printer_to_str(isl_set_get_ctx(domain));
      p_func = isl_printer_print_str(p_func, "(");
      loc++;
      int dim_cnt = 0;
      while((ch = c_text[loc]) && (dim_cnt < iter_num)) {
        if (ch == ']') {
          dim_cnt++;
          if (c_text[loc + 1] == '[')
            p_func = isl_printer_print_str(p_func, ", ");
          else
            p_func = isl_printer_print_str(p_func, ")");
        } else if (ch == '[') {
          loc++;
          continue;
        } else {
          char ch_str[2];
          ch_str[0] = ch;
          ch_str[1] = '\0';
          p_func = isl_printer_print_str(p_func, ch_str);
        }
        loc++;
      }

      char *func_str = isl_printer_get_str(p_func);
      p = isl_printer_print_str(p, func_str);
      if (at_LHS) {
        p_LHS = isl_printer_print_str(p_LHS, func_str);
        LHS_func = isl_printer_get_str(p_LHS);
        at_LHS = 0;
      }
     
      free(func_str);
      isl_printer_free(p_func);
    } 
    char ch_str[2];
    ch_str[0] = ch;
    ch_str[1] = '\0';
    p = isl_printer_print_str(p, ch_str);
    if (at_LHS) {
      p_LHS = isl_printer_print_str(p_LHS, ch_str);
    }
    loc++;  
  }

  t2s_text = isl_printer_get_str(p);
  isl_printer_free(p);
  isl_printer_free(p_LHS);
  free(iter_domain);
  free(LHS_func);
  free(c_text);
  isl_set_free(domain);

  return t2s_text;
}

/* Generate the number denotes how many times the given function has been updated. */
static int get_t2s_URE_update_level(struct t2s_URE **UREs, int URE_num, __isl_take char *func_name) {
  char **URE_names = NULL;
  if (URE_num > 0) {
    URE_names = (char **)malloc(URE_num * sizeof(char *));
    for (int i = 0; i < URE_num; i++) {
      URE_names[i] = strdup(UREs[i]->name);
    }
  }
  int update_level = -1;
  for (int i = 0; i < URE_num; i++) {
    char *cur_name = URE_names[i];
    if (strlen(cur_name) >= strlen(func_name)) {
      char cur_name_prefix[strlen(cur_name) + 1];
      char ch;
      int loc = 0;
      while ((ch = cur_name[loc]) != '\0') {
        if (ch == '.')
          break;
        else {
          cur_name_prefix[loc] = cur_name[loc];
          loc++;
        }
      }
      cur_name_prefix[loc] = '\0';
      if (!strcmp(cur_name_prefix, func_name))
        update_level++;
    }
  }
   
  if (URE_num > 0) {
    for (int i = 0; i < URE_num; i++) {
      free(URE_names[i]);
    }
  }
  free(URE_names);
  free(func_name);

  return update_level;
}

/* Given the func name, update the URE name and the update_level. */
static __isl_give struct t2s_URE *create_t2s_URE(__isl_keep struct t2s_URE **UREs, int URE_num, __isl_take char *func_name, __isl_take char *URE_text, int d, isl_ctx *ctx) {
  struct t2s_URE *URE = (struct t2s_URE *)malloc(sizeof(struct t2s_URE));

  char **URE_names = NULL;
  if (URE_num > 0) {
    URE_names = (char **)malloc(URE_num * sizeof(char *));
    for (int i = 0; i < URE_num; i++) {
      URE_names[i] = strdup(UREs[i]->name);
    }
  }
  int update_level = -1;
  for (int i = 0; i < URE_num; i++) {
    char *cur_name = URE_names[i];
    if (strlen(cur_name) >= strlen(func_name)) {
      char cur_name_prefix[strlen(cur_name) + 1];
      char ch;
      int loc = 0;
      while ((ch = cur_name[loc]) != '\0') {
        if (ch == '.')
          break;
        else {
          cur_name_prefix[loc] = cur_name[loc];
          loc++;
        }
      }
      cur_name_prefix[loc] = '\0';
      if (!strcmp(cur_name_prefix, func_name))
        update_level++;
    }
  }
  
  isl_printer *p = isl_printer_to_str(ctx);
  p = isl_printer_print_str(p, func_name);
  if (update_level >= 0) {
    p = isl_printer_print_str(p, ".update(");
    p = isl_printer_print_int(p, update_level);
    p = isl_printer_print_str(p, ")");
  }
  URE->name = isl_printer_get_str(p);
  isl_printer_free(p);

  URE->d = d;
  URE->text = URE_text;
  URE->update_level = update_level;

  if (URE_num > 0) {
    for (int i = 0; i < URE_num; i++) {
      free(URE_names[i]);
    }
  }
  free(URE_names);
  free(func_name);

  return URE;
}

/* Create the T2S URE from the statement text. */
static isl_stat create_t2s_URE_from_text(struct t2s_data *data, __isl_take char *URE_text, int d, isl_ctx *ctx) {
  char *func_name; 
  char ch;
  int loc = 0;
  isl_printer *p = isl_printer_to_str(ctx);  
  char *func_name_tmp;
  struct t2s_URE **UREs = data->URE;
  int URE_num = data->URE_num;

  while ((ch = URE_text[loc]) != '\0') {
    if (ch == '=')
      break;
    char ch_arr[2];
    ch_arr[0] = ch;
    ch_arr[1] = '\0';
    p = isl_printer_print_str(p, ch_arr);
    loc++;
  }
  func_name_tmp = isl_printer_get_str(p);
  isl_printer_free(p);
   
  loc = strlen(func_name_tmp) - 1;
  while((ch = func_name_tmp[loc--]) != ' ') {
    ;
  }
  char *func_decl = (char *)malloc(sizeof(char) * (loc + 1 + 1));
  strncpy(func_decl, func_name_tmp, loc + 1);
  func_decl[loc + 1] = '\0';

  while((ch = func_name_tmp[loc--]) != '(') {
    ;
  }
  func_name = (char *)malloc(sizeof(char) * (loc + 1 + 1));
  strncpy(func_name, func_name_tmp, loc + 1);
  func_name[loc + 1] = '\0';
  free(func_name_tmp);

  int update_level = get_t2s_URE_update_level(UREs, URE_num, strdup(func_name));
//  if (update_level == -1) {
//    p = isl_printer_to_str(ctx);
//    p = isl_printer_print_str(p, func_decl);
//    p = isl_printer_print_str(p, " = 0;\n");
//    char *init_URE_text = isl_printer_get_str(p);
//    isl_printer_free(p);
//
//    data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
//    data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(func_name), init_URE_text, d, ctx);
//    data->URE_num++;
//  }

  /* Add the statement URE. */
  data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
  data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(func_name), URE_text, d, ctx);
  data->URE_num++;
  
  free(func_name);
  free(func_decl);
  return isl_stat_ok;
}

/* Free up the t2s_stmt_data. */
static __isl_null struct t2s_stmt_data *t2s_stmt_data_free(__isl_take struct t2s_stmt_data *d) {
  if (!d)
    return NULL;

  for (int i = 0; i < d->stmt_num; i++) {
    ppcg_stmt_free(d->stmts[i]);
    isl_set_free(d->stmt_domain[i]);
  }
  free(d->stmts);
  free(d->stmt_domain);

  isl_set_free(d->stmt_anchor_domain);
  isl_pw_multi_aff_free(d->iterator_map);

  for (int i = 0; i < d->n_acc_group; i++) {
    for (int j = 0; j < d->n_dep_per_acc_group[i]; j++) {
      polysa_dep_free(d->stmt_deps[i][j]);
    }
    free(d->stmt_deps[i]);    
  }
  free(d->stmt_deps);
  free(d->n_dep_per_acc_group);

  free(d);
  
  return NULL;
}

/* Buggy Implementation. */
///* For each user statement, there will be multiple T2S UREs generated given different
// * dependences. To improve the hardware efficiency and code readability, there UREs
// * will be merged into one UREs in this function. 
// * For example, given two UREs:
// * A(i, j, k) = select(D1, A(i, j, k) + B(i, j, k));
// * A(i, j, k) = select(D2, A(i, j, k - 1) + B(i, j, k));
// * We will merge them into one URE below:
// * A(i, j, k) = select(D1, A(i, j, k), select(D2, A(i, j, k - 1))) + 
// *              select(D1, B(i, j, k), select(D2, B(i, j, k)));
// */
//char *merge_t2s_stmt_text(__isl_take char **stmt_texts, int n, isl_ctx *ctx) {
//  char **iter_domain = (char **)malloc(sizeof(char *) * n);
//  int n_func = 0;
//  
//  /* Collect number of functions in the statement. */
//  char *text = stmt_texts[0];
//  char ch;
//  int loc = 0;
//  bool is_func = true;
//  while((ch = text[loc]) != '\0') {
//    if (ch == '(') {
//      ch = text[++loc];
//      while(ch != '(' && ch != ')') {
//        ch = text[++loc];
//        if (ch == '=' || ch == '>' || ch == '<')
//          is_func = false;
//      }
//      if (ch == ')') {
//        if (is_func)
//          n_func++;      
//        is_func = true;
//      } else if (ch == '(') {
//        loc--;
//      }
//    }
//    loc++;
//  }
//  
//  char ***func = (char ***)malloc(sizeof(char **) * n_func);
//  for (int i = 0; i < n_func; i++) {
//    func[i] = (char **)malloc(sizeof(char *) * n);
//  }
//  int* func_offset = (int *)malloc(sizeof(int) * n_func);
//
//  /* Collect all the iteration domains and func names. */
//  for (int i = 0; i < n; i++) {
//    char *text = stmt_texts[i];
//    char ch;
//    int loc = 0;
//    int func_id = 0;
//    while((ch = text[loc]) != '\0') {      
//      if (ch == 's') {
//        char token[6];
//        if (loc + 6 <= strlen(text)) {
//          strncpy(token, text + loc, 6);
//        }
//        if (!strcmp(token, "select")) {
//          /* Collect the iteration domain. */
//          isl_printer *p_str = isl_printer_to_str(ctx);
//          loc += 6;
//          loc += 1;
//          while (ch = text[loc] != ',') {
//            char ch_arr[2];
//            ch_arr[0] = ch;
//            ch_arr[1] = '\0';
//            p_str = isl_printer_print_str(p_str, ch_arr);
//            loc++;
//          }
//          iter_domain[i] = isl_printer_get_str(p_str);
//          isl_printer_free(p_str);
//        }
//      }
// 
//      /* Collect the func names. */
//      if (ch == '(') {
//        while (loc >= 0 && ((ch = text[loc]) != ' ')) 
//          loc--;
//        loc++;
//        isl_printer *p_str = isl_printer_to_str(ctx);
//        while((ch = text[loc]) != '(') {
//          char ch_arr[2];
//          ch_arr[0] = ch;
//          ch_arr[1] = '\0';
//          p_str = isl_printer_print_str(p_str, ch_arr);
//          loc++;
//        }
//        // int loc_cur = loc;        
//        char ch_arr[2];
//        ch_arr[0] = ch;
//        ch_arr[1] = '\0';
//        p_str = isl_printer_print_str(p_str, ch_arr);
//        loc++;
//        while((ch = text[loc]) != ')') {
//          char ch_arr[2];
//          ch_arr[0] = ch;
//          ch_arr[1] = '\0';
//          p_str = isl_printer_print_str(p_str, ch_arr);
//          if (ch == '(') {
//            loc--;
//            p_str = isl_printer_free(p_str);
//            break;
//          }
//          loc++;
//        }
//        if (p_str) {
//          p_str = isl_printer_print_str(p_str, ")");        
//          func[func_id][i] = isl_printer_get_str(p_str);
//          if (i == 0)
//            func_offset[func_id] = loc - strlen(func[func_id][i]) + 1;
//          func_id++;
//          p_str = isl_printer_free(p_str);
//        }
//      }
//      loc++;
//    }
//  } 
//
//  /* Scan through the statement text and plug in the functions and domains. */
//  loc = 0;
//  text = stmt_texts[0];
//  isl_printer *p_str = isl_printer_to_str(ctx);
//  int func_cnt = 0;
//  while ((ch = text[loc]) != '\0') {
//    if (loc == func_offset[func_cnt]) {
//      if (func_cnt == 0)
//        p_str = isl_printer_print_str(p_str, func[func_cnt][0]);
//      else {
//        for (int i = 0; i < n; i++) {
//          if (i > 0)
//            p_str = isl_printer_print_str(p_str, ", ");
//          p_str = isl_printer_print_str(p_str, "select(");
//          p_str = isl_printer_print_str(p_str, iter_domain[i]);
//          p_str = isl_printer_print_str(p_str, ", ");
//          p_str = isl_printer_print_str(p_str, func[func_cnt][i]);
//        }
//        for (int i = 0; i < n; i++) {
//          p_str = isl_printer_print_str(p_str, ")");
//        }
//      }
//      loc += strlen(func[func_cnt][0]);
//      func_cnt++;
//    } else {
//      char ch_arr[2];
//      ch_arr[0] = ch;
//      ch_arr[1] = '\0';
//      p_str = isl_printer_print_str(p_str, ch_arr);
//    }
//    loc++;
//  }
//
//  char *merge_text = isl_printer_get_str(p_str);
//  isl_printer_free(p_str);
//  for (int i = 0; i < n; i++) {
//    free(stmt_texts[i]);    
//  }
//  free(stmt_texts);
//
//  return merge_text;
//}

/* Buggy implementation. */
///* Transform each user statement in the original program to a T2S URE 
// * w/ URE simplification. 
// * In this function, only one URE is generated for each user statement.
// */
//static __isl_give isl_schedule_node *gen_stmt_text_single(__isl_take isl_schedule_node *node, void *user)
//{
//  struct ppcg_stmt *stmt;
//  isl_set *domain;
//  isl_space *space;
//  isl_id *id;
//  struct t2s_data *data = user;
//  struct t2s_stmt_data *stmt_data;
// 
//  if (!node)
//    return NULL;
//
//  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
//    return node;
//
//  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
//
//  /* Find the stmt. */
//  stmt = isl_calloc_type(data->ctx, struct ppcg_stmt);
//  domain = isl_set_from_union_set(isl_schedule_node_get_domain(node));
//  space = isl_set_get_space(domain);
//  id = isl_space_get_tuple_id(space, isl_dim_set);
//  stmt->stmt = find_stmt(data->scop, id);  
//  isl_space_free(space);
//  isl_set_free(domain);
//
//  /* Decide if there will be multiple T2S stmts generated for one stmt. 
//   * Construct the unique acc->func mapping for each T2S stmts.*/
//  stmt_data = isl_calloc_type(data->ctx, struct t2s_stmt_data);
//  stmt_data->stmt_num = 0;
//  stmt_data->stmts = NULL;
//  stmt_data->stmt_domain = NULL;
//  stmt_data->stmt_deps = NULL;
//  stmt_data->n_acc_group = 0;
//  stmt_data->n_dep_per_acc_group = 0;
//  stmt_data->dep_stmt_pair = NULL;
//  stmt_data->iterator_map = NULL;
//  isl_set_list *stmt_domain_list = isl_union_set_get_set_list(data->stmt_domain);
//  for (int i = 0; i < isl_union_set_n_set(data->stmt_domain); i++) {
//    isl_set *stmt_domain_i = isl_set_list_get_set(stmt_domain_list, i);
//    isl_space *space_i = isl_set_get_space(stmt_domain_i);
//    isl_id *id_i = isl_space_get_tuple_id(space_i, isl_dim_set);
//    if (id_i == id)
//      stmt_data->stmt_anchor_domain = isl_set_copy(stmt_domain_i);    
//    isl_set_free(stmt_domain_i);
//    isl_space_free(space_i);
//    isl_id_free(id_i);
//  }
//  isl_set_list_free(stmt_domain_list);
//  isl_id_free(id);
//
//  data->stmt_data = stmt_data;
//  /* Extract the ref2expr for each access. */
//  extract_t2s_stmt_access(stmt, data);
//
//  data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + data->stmt_data->stmt_num));
//  char **stmt_texts = isl_calloc_array(data->ctx, char *, data->stmt_data->stmt_num);
//
//  /* Print the stmt to data->t2s_stmt_text and update data->t2s_stmt_num. */
//  for (int i = 0; i < data->stmt_data->stmt_num; i++) {
//    isl_printer *p_str = isl_printer_to_str(data->ctx);
//	  p_str = isl_printer_set_output_format(p_str, ISL_FORMAT_C);
//    struct ppcg_stmt *stmt_i = data->stmt_data->stmts[i];
//    p_str = pet_stmt_print_body(stmt_i->stmt, p_str, stmt_i->ref2expr);
//    char *stmt_text = isl_printer_get_str(p_str);
//    stmt_texts[i] = c_to_t2s_stmt(stmt_text, isl_set_copy(data->stmt_data->stmt_domain[i]), data->iter_num);
////    create_t2s_URE_from_text(data, stmt_text, 0, ctx);      
//    isl_printer_free(p_str);
//  }
//  char *merged_stmt_text = merge_t2s_stmt_text(stmt_texts, data->stmt_data->stmt_num, data->ctx);
//  create_t2s_URE_from_text(data, merged_stmt_text, 0, ctx);
//
//  data->stmt_data = t2s_stmt_data_free(stmt_data);
//
//  return node;
//}

/* Transform each user statement in the original program to a T2S URE. 
 * w/o URE simplification.
 * In this function, multiple UREs can be generated for each user statement.
 */
static __isl_give isl_schedule_node *gen_stmt_text(__isl_take isl_schedule_node *node, void *user)
{
  struct ppcg_stmt *stmt;
  isl_set *domain;
  isl_space *space;
  isl_id *id;
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data;
 
  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Find the stmt. */
  stmt = isl_calloc_type(data->ctx, struct ppcg_stmt);
  domain = isl_set_from_union_set(isl_schedule_node_get_domain(node));
  space = isl_set_get_space(domain);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  stmt->stmt = find_pet_stmt(data->scop, id);  
  isl_space_free(space);
  isl_set_free(domain);

  /* Decide if there will be multiple T2S stmts generated for one stmt. 
   * Construct the unique acc->func mapping for each T2S stmts.*/
  stmt_data = isl_calloc_type(data->ctx, struct t2s_stmt_data);
  stmt_data->stmt_num = 0;
  stmt_data->stmts = NULL;
  stmt_data->stmt_domain = NULL;
  stmt_data->stmt_deps = NULL;
  stmt_data->n_acc_group = 0;
  stmt_data->n_dep_per_acc_group = 0;
  stmt_data->dep_stmt_pair = NULL;
  stmt_data->iterator_map = NULL;
  isl_set_list *stmt_domain_list = isl_union_set_get_set_list(data->stmt_domain);
  for (int i = 0; i < isl_union_set_n_set(data->stmt_domain); i++) {
    isl_set *stmt_domain_i = isl_set_list_get_set(stmt_domain_list, i);
    isl_space *space_i = isl_set_get_space(stmt_domain_i);
    isl_id *id_i = isl_space_get_tuple_id(space_i, isl_dim_set);
    if (id_i == id)
      stmt_data->stmt_anchor_domain = isl_set_copy(stmt_domain_i);    
    isl_set_free(stmt_domain_i);
    isl_space_free(space_i);
    isl_id_free(id_i);
  }
  isl_set_list_free(stmt_domain_list);
  isl_id_free(id);

//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_print_set(p, stmt_data->stmt_anchor_domain);
//  printf("\n");
//  // debug

  data->stmt_data = stmt_data;
  /* Extract the ref2expr for each access. */
  extract_t2s_stmt_access(stmt, data);

  data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + data->stmt_data->stmt_num));
  /* Print the stmt to data->t2s_stmt_text and update data->t2s_stmt_num. */
  for (int i = 0; i < data->stmt_data->stmt_num; i++) {
    isl_printer *p_str = isl_printer_to_str(data->ctx);
	  p_str = isl_printer_set_output_format(p_str, ISL_FORMAT_C);
    struct ppcg_stmt *stmt_i = data->stmt_data->stmts[i];
    p_str = pet_stmt_print_body(stmt_i->stmt, p_str, stmt_i->ref2expr);
//    // debug
//    isl_printer *p_debug = isl_printer_to_file(data->ctx, stdout);
//    p_debug = isl_printer_print_set(p_debug, data->stmt_data->stmt_domain[i]);
//    printf("\n");
//    // debug
    char *stmt_text = isl_printer_get_str(p_str);
    stmt_text = c_to_t2s_stmt(stmt_text, isl_set_copy(data->stmt_data->stmt_domain[i]), data->iter_num);
    create_t2s_URE_from_text(data, stmt_text, 0, ctx);  
    isl_printer_free(p_str);
  }

  data->stmt_data = t2s_stmt_data_free(stmt_data);

  return node;
}

/* Generate the array access from isl_multi_pw_aff "mpa". */
static __isl_give char *array_acc_from_multi_pw_aff(__isl_take isl_multi_pw_aff *mpa)
{
  isl_ctx *ctx;

  ctx = isl_multi_pw_aff_get_ctx(mpa);
  isl_space *space = isl_multi_pw_aff_get_space(mpa);
  const char *array_name = isl_space_get_tuple_name(space, isl_dim_out);
  isl_printer *p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_multi_pw_aff(p_str, mpa);
  char *mpa_str = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  p_str = isl_printer_to_str(ctx);
  p_str = isl_printer_print_str(p_str, array_name);
  char ch;
  int loc = 0;
  while ((ch = mpa_str[loc]) != '\0') {
    if (ch == '(') {
      while((ch = mpa_str[++loc]) == '(')
        ;
      loc--;
      p_str = isl_printer_print_str(p_str, "[");
      while((ch = mpa_str[++loc]) != ')') {
        char ch_arr[2];
        ch_arr[0] = ch;
        ch_arr[1] = '\0';
        p_str = isl_printer_print_str(p_str, ch_arr);
      }
      p_str = isl_printer_print_str(p_str, "]");
    }
    loc++;
  }
  
  char *acc_str = isl_printer_get_str(p_str);
  free(mpa_str);
  isl_printer_free(p_str);
  isl_multi_pw_aff_free(mpa);
  isl_space_free(space);

  return acc_str;
}

/* Set the iterator names in the target domain. */
static __isl_give isl_set *t2s_set_set_iters(__isl_take isl_set *s)
{
  isl_ctx *ctx = isl_set_get_ctx(s);
  for (int i = 0; i < isl_set_dim(s, isl_dim_set); i++) {
    char iter_name[100];
    sprintf(iter_name, "c%d", i);
    isl_id *id_i = isl_id_alloc(ctx, iter_name, NULL);
    s = isl_set_set_dim_id(s, isl_dim_set, i, id_i);
  }
  
  return s;
}

/* Set up the iterator names. */
static __isl_give isl_multi_pw_aff *t2s_set_multi_pw_aff_iters(__isl_take isl_multi_pw_aff *mpa)
{
  isl_ctx *ctx = isl_multi_pw_aff_get_ctx(mpa);
  for (int i = 0; i < isl_multi_pw_aff_dim(mpa, isl_dim_in); i++) {
    char iter_name[100];
    sprintf(iter_name, "c%d", i);
    isl_id *id_i = isl_id_alloc(ctx, iter_name, NULL);
    mpa = isl_multi_pw_aff_set_dim_id(mpa, isl_dim_in, i, id_i);
  }

  return mpa;
}

/* Create UREs for live-in accesses associated with RAR deps. */
static int t2s_rar_URE_access(__isl_keep pet_expr *expr, void *user)
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data = data->stmt_data;
  isl_multi_pw_aff *index;
  isl_id *id;
  id = isl_id_copy(expr->acc.ref_id);
  index = isl_multi_pw_aff_copy(expr->acc.index);
  struct polysa_dep *dep;
  int n;
  isl_ctx *ctx = data->ctx;
  char *URE_text;

  for (n = 0; n < data->ndeps; n++) {
    dep = data->deps[n];
    if (dep->dest == id && dep->type == POLYSA_DEP_RAR)
      break;
  }

  if (n != data->ndeps) {
    isl_set *stmt_domain = isl_set_copy(stmt_data->stmt_anchor_domain);
    isl_set *dep_dest_domain = isl_set_copy(dep->dest_sched_domain);

    /* Generate the init domain */
    isl_set *init_domain = isl_set_subtract(stmt_domain, isl_set_copy(dep_dest_domain));

    isl_set *anchor_domain = isl_set_copy(data->anchor_domain);
    anchor_domain = isl_set_set_tuple_name(anchor_domain, isl_set_get_tuple_name(init_domain));
    init_domain = isl_set_gist(init_domain, isl_set_copy(anchor_domain));
    isl_set *reuse_domain = isl_set_gist(dep_dest_domain, anchor_domain);
    
    /* Peel off the scalar dimensions */
    init_domain = t2s_peel_off_scalar_dims(init_domain, data->schedule);
    reuse_domain = t2s_peel_off_scalar_dims(reuse_domain, data->schedule);
      
    /* Set up the iterator names. */
    init_domain = t2s_set_set_iters(init_domain);
    reuse_domain = t2s_set_set_iters(reuse_domain);

    char *init_domain_str = isl_set_to_t2s_format(init_domain);
    isl_set_free(init_domain);

    char *reuse_domain_str = isl_set_to_t2s_format(reuse_domain);
    isl_set_free(reuse_domain);

    /* Generate the func name .*/
    isl_id *func = isl_id_to_id_get(data->ref2func, isl_id_copy(id));
    isl_printer *p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    char *func_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);
  
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    p_str = isl_printer_print_str(p_str, "(");
    for (int i = 0; i < data->iter_num; i++) {
      if (i > 0) {
        p_str = isl_printer_print_str(p_str, ", ");
      }
      char iter_name[100];
      sprintf(iter_name, "c%d", i);
      p_str = isl_printer_print_str(p_str, iter_name);
    }
    p_str = isl_printer_print_str(p_str, ")");
    char *func_str = isl_printer_get_str(p_str);

    isl_printer_free(p_str);
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    p_str = isl_printer_print_str(p_str, "(");
    for (int i = 0; i < data->iter_num; i++) {
      if (i > 0) {
        p_str = isl_printer_print_str(p_str, ", ");
      }
      char iter_name[100];
      isl_val *ele = isl_vec_get_element_val(dep->disvec, i);
      if (isl_val_is_zero(ele)) {
        sprintf(iter_name, "c%d", i);
      } else {
        sprintf(iter_name, "c%d - %ld", i, isl_val_get_num_si(ele));
      }
      p_str = isl_printer_print_str(p_str, iter_name);
      isl_val_free(ele);
    }
    p_str = isl_printer_print_str(p_str, ")");
    char *reuse_func_str = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    /* Generate the transformed index. */
    isl_multi_pw_aff *trans_index = isl_multi_pw_aff_copy(index);
    trans_index = isl_multi_pw_aff_pullback_pw_multi_aff(trans_index, isl_pw_multi_aff_copy(stmt_data->iterator_map));

    /* Set up the iterator names. */
    trans_index = t2s_set_multi_pw_aff_iters(trans_index);
    char *acc_str = array_acc_from_multi_pw_aff(trans_index);

    /* Generate the URE. */
    int update_level = get_t2s_URE_update_level(data->URE, data->URE_num, strdup(func_name));

    /* Comment out. The latest T2S no longer requires this. */
//    if (update_level == -1) {
//      p_str = isl_printer_to_str(ctx);
//      p_str = isl_printer_print_str(p_str, func_str);
//      p_str = isl_printer_print_str(p_str, " = 0;\n");
//      URE_text = isl_printer_get_str(p_str);
//      isl_printer_free(p_str);
//
//      data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
//      data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(func_name), URE_text, 0, ctx); 
//      data->URE_num++;
//    }

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, func_str);
    p_str = isl_printer_print_str(p_str, " = ");
    p_str = isl_printer_print_str(p_str, "select(");
    p_str = isl_printer_print_str(p_str, init_domain_str);
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, acc_str);
    p_str = isl_printer_print_str(p_str, ", select(");
    p_str = isl_printer_print_str(p_str, reuse_domain_str);
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, reuse_func_str);
//    p_str = isl_printer_print_str(p_str, ", ");
//    p_str = isl_printer_print_str(p_str, func_str);
    p_str = isl_printer_print_str(p_str, "));\n");
    URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

//    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
//    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
//    data->t2s_stmt_num++;

    data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
    data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(func_name), URE_text, 0, ctx);
    data->URE_num++;

    isl_id_free(func);
    free(func_str);
    free(func_name);
    free(init_domain_str);
    free(acc_str);
    free(reuse_domain_str);
    free(reuse_func_str);
  }

  isl_id_free(id);
  isl_multi_pw_aff_free(index);

  return 0;
}

/* Extract UREs for live-in accesses. */
static isl_stat extract_rar_URE(__isl_take struct ppcg_stmt *stmt, struct t2s_data *data) {
  pet_tree_foreach_access_expr(stmt->stmt->body, &t2s_rar_URE_access, data);
  ppcg_stmt_free(stmt);

  return isl_stat_ok;
}

/* Generate the reuse (RAR) UREs for the operands in the user statements. */
static __isl_give isl_schedule_node *gen_op_stmt_text(__isl_take isl_schedule_node *node, void *user) 
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data;
  struct ppcg_stmt *stmt;
  isl_set *domain;
  isl_space *space;
  isl_id *id;

  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  /* Find the statement. */
  stmt = isl_calloc_type(data->ctx, struct ppcg_stmt);
  domain = isl_set_from_union_set(isl_schedule_node_get_domain(node));
  space = isl_set_get_space(domain);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  stmt->stmt = find_pet_stmt(data->scop, id);
  isl_space_free(space);
  isl_set_free(domain);

  stmt_data = isl_calloc_type(data->ctx, struct t2s_stmt_data);
  stmt_data->stmt_num = 0;
  stmt_data->stmts = NULL;
  stmt_data->stmt_domain = NULL;
  stmt_data->stmt_deps = NULL;
  stmt_data->n_acc_group = 0;
  stmt_data->n_dep_per_acc_group = 0;
  stmt_data->dep_stmt_pair = NULL;
  stmt_data->iterator_map = NULL;
  isl_set_list *stmt_domain_list = isl_union_set_get_set_list(data->stmt_domain);
  for (int i = 0; i < isl_union_set_n_set(data->stmt_domain); i++) {
    isl_set *stmt_domain_i = isl_set_list_get_set(stmt_domain_list, i);
    isl_space *space_i = isl_set_get_space(stmt_domain_i);
    isl_id *id_i = isl_space_get_tuple_id(space_i, isl_dim_set);
    if (id_i == id)
      stmt_data->stmt_anchor_domain = isl_set_copy(stmt_domain_i);
    isl_set_free(stmt_domain_i);
    isl_space_free(space_i);
    isl_id_free(id_i);
  }
  isl_set_list_free(stmt_domain_list);
  isl_id_free(id);

  data->stmt_data = stmt_data;

  /* Extract the schedule. */
  isl_map *sched = isl_map_from_union_map(isl_schedule_node_get_prefix_schedule_relation(node));
  sched = isl_map_reverse(sched);
  isl_pw_multi_aff *iterator_map = isl_pw_multi_aff_from_map(sched);

  data->stmt_data->iterator_map = iterator_map;

  extract_rar_URE(stmt, data);

  data->stmt_data = t2s_stmt_data_free(stmt_data);

  return node;
}

/* Create UREs for live-out accesses. */
static int t2s_drain_URE_access(__isl_keep pet_expr *expr, void *user)
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data = data->stmt_data;
  isl_multi_pw_aff *index;
  isl_id *id;
  id = isl_id_copy(expr->acc.ref_id);
  index = isl_multi_pw_aff_copy(expr->acc.index);
  struct polysa_dep *dep;
  int n;
  isl_ctx *ctx = data->ctx;
  isl_set *writeout_domain = isl_set_copy(stmt_data->stmt_anchor_domain);
  int is_drain = 0;

  for (n = 0; n < data->ndeps; n++) {
    dep = data->deps[n];
    if (dep->src == id && dep->type == POLYSA_DEP_WAW) {
      isl_set *dep_src_domain = isl_set_copy(dep->src_sched_domain);

      /* Generate the writeout domain */
      writeout_domain = isl_set_subtract(writeout_domain, dep_src_domain);
      is_drain = 1;
    }
  }

  if (isl_set_is_empty(writeout_domain) || !is_drain) {
    isl_set_free(writeout_domain);
    isl_id_free(id);
    isl_multi_pw_aff_free(index);
    return 0;
  } else {
    isl_set *anchor_domain = isl_set_copy(data->anchor_domain);
    anchor_domain = isl_set_set_tuple_name(anchor_domain, isl_set_get_tuple_name(writeout_domain));
    writeout_domain = isl_set_gist(writeout_domain, anchor_domain);

    /* Peel off the scalar dimensions. */
    writeout_domain = t2s_peel_off_scalar_dims(writeout_domain, data->schedule);

    /* Set up the iterator names. */
    writeout_domain = t2s_set_set_iters(writeout_domain);
    char *writeout_domain_str = isl_set_to_t2s_format(writeout_domain);
    isl_set_free(writeout_domain);

    /* Generate the func name .*/
    isl_id *func = isl_id_to_id_get(data->ref2func, isl_id_copy(id));
    isl_printer *p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    p_str = isl_printer_print_str(p_str, "_drain");
    char *drain_func_name = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    p_str = isl_printer_print_str(p_str, "(");
    for (int i = 0; i < data->iter_num; i++) {
      if (i > 0) {
        p_str = isl_printer_print_str(p_str, ", ");
      }
      char iter_name[100];
      sprintf(iter_name, "c%d", i);
      p_str = isl_printer_print_str(p_str, iter_name);
    }
    p_str = isl_printer_print_str(p_str, ")");
    char *func_str = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_id(p_str, func);
    p_str = isl_printer_print_str(p_str, "_drain(");
    for (int i = 0; i < data->iter_num; i++) {
      if (i > 0) {
        p_str = isl_printer_print_str(p_str, ", ");
      }
      char iter_name[100];
      sprintf(iter_name, "c%d", i);
      p_str = isl_printer_print_str(p_str, iter_name);
    }
    p_str = isl_printer_print_str(p_str, ")");
    char *drain_func_str = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

//    /* Generate the transformed index. */
//    isl_multi_pw_aff *trans_index = isl_multi_pw_aff_copy(index);
//    trans_index = isl_multi_pw_aff_pullback_pw_multi_aff(trans_index, isl_pw_multi_aff_copy(stmt_data->iterator_map));
//
//    /* Set up the iterator names. */
//    trans_index = t2s_set_multi_pw_aff_iters(trans_index);
//    char *acc_str = array_acc_from_multi_pw_aff(trans_index);

    /* Generate the URE. */
//    p_str = isl_printer_to_str(ctx);
//    p_str = isl_printer_print_str(p_str, drain_func_str);    
//    p_str = isl_printer_print_str(p_str, " = 0;\n");
//    char *URE_text = isl_printer_get_str(p_str);
//    isl_printer_free(p_str);

//    data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
//    data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(drain_func_name), URE_text, 1, ctx);
//    data->URE_num++;

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, drain_func_str);
    p_str = isl_printer_print_str(p_str, " = ");
    p_str = isl_printer_print_str(p_str, "select(");
    p_str = isl_printer_print_str(p_str, writeout_domain_str);
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, func_str);
//    p_str = isl_printer_print_str(p_str, ", ");
//    p_str = isl_printer_print_str(p_str, drain_func_str);
    p_str = isl_printer_print_str(p_str, ");\n");
    char *URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

//    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
//    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
//    data->t2s_stmt_num++;

    data->URE = (struct t2s_URE **)realloc(data->URE, sizeof(struct t2s_URE *) * (data->URE_num + 1));
    data->URE[data->URE_num] = create_t2s_URE(data->URE, data->URE_num, strdup(drain_func_name), URE_text, 1, ctx);
    data->URE_num++;

    isl_id_free(func);
    free(drain_func_name);
    free(func_str);
    free(drain_func_str);
    free(writeout_domain_str);
//    free(acc_str);
  }

  isl_id_free(id);
  isl_multi_pw_aff_free(index);

  return 0;
 
}

/* Generate UREs for live-out accesses. */
static isl_stat extract_drain_URE(__isl_take struct ppcg_stmt *stmt, struct t2s_data *data) {
  pet_tree_foreach_access_expr(stmt->stmt->body, &t2s_drain_URE_access, data);
  ppcg_stmt_free(stmt);

  return isl_stat_ok;
}

/* Generate the drain UREs for the intermediate variables in the user statement. */
static __isl_give isl_schedule_node *gen_drain_stmt_text(__isl_take isl_schedule_node *node, void *user)
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data;
  struct ppcg_stmt *stmt;
  isl_set *domain;
  isl_space *space;
  isl_id *id;

  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  /* Find the statement. */
  stmt = isl_calloc_type(data->ctx, struct ppcg_stmt);
  domain = isl_set_from_union_set(isl_schedule_node_get_domain(node));
  space = isl_set_get_space(domain);
  id = isl_space_get_tuple_id(space, isl_dim_set);
  stmt->stmt = find_pet_stmt(data->scop, id);
  isl_space_free(space);
  isl_set_free(domain);

  stmt_data = isl_calloc_type(data->ctx, struct t2s_stmt_data);
  stmt_data->stmt_num = 0;
  stmt_data->stmts = NULL;
  stmt_data->stmt_domain = NULL;
  stmt_data->stmt_deps = NULL;
  stmt_data->n_acc_group = 0;
  stmt_data->n_dep_per_acc_group = 0;
  stmt_data->dep_stmt_pair = NULL;
  stmt_data->iterator_map = NULL;
  isl_set_list *stmt_domain_list = isl_union_set_get_set_list(data->stmt_domain);
  for (int i = 0; i < isl_union_set_n_set(data->stmt_domain); i++) {
    isl_set *stmt_domain_i = isl_set_list_get_set(stmt_domain_list, i);
    isl_space *space_i = isl_set_get_space(stmt_domain_i);
    isl_id *id_i = isl_space_get_tuple_id(space_i, isl_dim_set);
    if (id_i == id) {
      stmt_data->stmt_anchor_domain = isl_set_copy(stmt_domain_i);
    }
    isl_set_free(stmt_domain_i);
    isl_space_free(space_i);
    isl_id_free(id_i);      
  }
  isl_set_list_free(stmt_domain_list);
  isl_id_free(id);

  data->stmt_data = stmt_data;

  /* Extract the schedule. */
  isl_map *sched = isl_map_from_union_map(isl_schedule_node_get_prefix_schedule_relation(node));
  sched = isl_map_reverse(sched);
  isl_pw_multi_aff *iterator_map = isl_pw_multi_aff_from_map(sched);

  data->stmt_data->iterator_map = iterator_map;

  extract_drain_URE(stmt, data);

  data->stmt_data = t2s_stmt_data_free(stmt_data);

  return node;
}

/* Print UREs in T2S code. */
static __isl_give isl_schedule *gen_stmt_text_wrap(__isl_take isl_schedule *schedule, struct t2s_data *data)
{
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_print_str(data->p, "// UREs");
  data->p = isl_printer_end_line(data->p);

  /* Generate the reuse (RAR) statement. */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_op_stmt_text, data);

  /* Traverse each statmenet, build the ppcg_stmt struct and update
   * the ref2expr using T2S functions.
   * Print the stmt to data->t2s_stmt_text.
   */
  /* Option 1: Generate multiple UREs for one statement. */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_stmt_text, data);
//  /* Option 2: Generate single URE for one statement. (Buggy)*/
//  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
//    &gen_stmt_text_single, data);

  /* Generate the drain statement. */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_drain_stmt_text, data);

//  /* Print out the T2S stmt texts. */
//  for (int i = 0; i < data->t2s_stmt_num; i++) {
////    data->p = isl_printer_start_line(data->p);
//    data->p = isl_printer_print_str(data->p, data->t2s_stmt_text[i]);
////    data->p = isl_printer_end_line(data->p);
//  }
  /* Print out the URE texts. */
  for (int i = 0; i < data->URE_num; i++) {
    data->p = isl_printer_print_str(data->p, data->URE[i]->text);
  }
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_end_line(data->p);

  return schedule;
}

/* Extract the detailed information of iterators in the code, including:
 * - iterator name
 * - lower bound
 * - upper bound
 * - stride
 */
static isl_stat extract_iters(__isl_keep isl_schedule *schedule, struct t2s_data *data) {
  isl_ctx *ctx;
  isl_set *domain = isl_set_copy(data->anchor_domain); 
  int iter_num = data->iter_num;
  ctx = isl_set_get_ctx(domain);
  
//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_set(p, domain);
//  printf("\n");
//  // debug

  /* Peel off the scalar dimensions. */
  domain = t2s_peel_off_scalar_dims(domain, schedule);

//  // debug
////  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_set(p, domain);
//  printf("\n");
//  // debug

  data->iter = (struct polysa_iter **)malloc(sizeof(struct polysa_iter *) * iter_num);

  isl_map *domain_map = isl_map_from_range(isl_set_copy(domain));
  isl_fixed_box *box = isl_map_get_range_simple_fixed_box_hull(domain_map);
  isl_multi_aff *offset;
  isl_multi_val *size;
  isl_map_free(domain_map);

  if (isl_fixed_box_is_valid(box)) {
    offset = isl_fixed_box_get_offset(box);
    size = isl_fixed_box_get_size(box);
//    // debug
//    p = isl_printer_print_multi_aff(p, offset);
//    printf("\n");
//    p = isl_printer_print_multi_val(p, size);
//    printf("\n");
//    // debug
  }

  for (int i = 0; i < data->iter_num; i++) {
    struct polysa_iter *iter = (struct polysa_iter *)malloc(sizeof(struct polysa_iter));
    /* Stride. */
    isl_stride_info *si;
    si = isl_set_get_stride_info(domain, i);
    isl_val *s = isl_stride_info_get_stride(si);
//    // debug
//    p = isl_printer_print_val(p, s);
//    printf("\n");
//    // debug
    iter->stride = isl_val_get_num_si(s);
    isl_val_free(s);
    isl_stride_info_free(si);

    /* Name. */
    char iter_name[100];
    sprintf(iter_name, "c%d", i);
    iter->name = strdup(iter_name);

    char ts_iter_name[100];
    if (data->prog->type == POLYSA_SA_TYPE_ASYNC) {
      if (i < data->prog->space_w) {
        sprintf(ts_iter_name, "sloop%d", i);
      } else {
        sprintf(ts_iter_name, "tloop%d", i - data->prog->space_w);
      }
    } else if (data->prog->type == POLYSA_SA_TYPE_SYNC) {
      if (i < data->prog->time_w) {
        sprintf(ts_iter_name, "tloop%d", i);
      } else {
        sprintf(ts_iter_name, "sloop%d", i - data->prog->time_w);
      }
    }
    iter->ts_name = strdup(ts_iter_name);

    /* Bounds. */
    if (isl_fixed_box_is_valid(box)) {
      isl_aff *offset_i = isl_multi_aff_get_aff(offset, i);
      iter->lb = isl_aff_copy(offset_i);
//    // debug
//    p = isl_printer_print_aff(p, offset_i);
//    printf("\n");
//    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
//    p = isl_printer_print_aff(p, offset_i);
//    printf("\n");
//    // debug
      isl_val *size_i = isl_multi_val_get_val(size, i);
      offset_i = isl_aff_add_constant_val(offset_i, size_i);
      offset_i = isl_aff_add_constant_si(offset_i, -1);
      iter->ub = offset_i;
    }

    data->iter[i] = iter;
  }

  if (isl_fixed_box_is_valid(box)) {
    isl_multi_aff_free(offset);
    isl_multi_val_free(size);
  }
  isl_fixed_box_free(box);
  isl_set_free(domain);

  return isl_stat_ok;
}

/* Extract the dependence (RAW, RAR, WAW) from the program. */
static __isl_give isl_schedule *extract_deps(__isl_take isl_schedule *schedule, struct t2s_data *data) {
  isl_schedule_node *band;
  isl_union_map *dep_flow;
  isl_union_map *dep_rar;
  isl_union_map *dep_waw;
  isl_union_map *dep_total;
  isl_basic_map_list *deps;
  int ndeps;
  isl_basic_map *dep_i;
  struct polysa_dep *p_dep_i;
  isl_vec *disvec;

//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule(p, schedule);
//  printf("\n");
//  // debug
  if (data->scop->options->t2s_tile && data->scop->options->t2s_tile_phase == 1) {
    band = isl_schedule_get_root(schedule);
    band = isl_schedule_node_child(band, 0);
  } else {
    band = get_outermost_permutable_node(schedule);
  }
//  // debug
//  p = isl_printer_print_schedule_node(p, band);
//  printf("\n");
//  // debug
  dep_flow = data->scop->tagged_dep_flow;
  dep_rar = data->scop->tagged_dep_rar;
  dep_waw = data->scop->tagged_dep_waw;

  /* Add RAW deps. */
  deps = isl_union_map_get_basic_map_list(dep_flow);
  ndeps = isl_union_map_n_basic_map(dep_flow);
  data->ndeps = ndeps;
  data->deps = (struct polysa_dep **)malloc(data->ndeps * sizeof(struct polysa_dep *));

  for (int i = 0; i < ndeps; i++) {
    p_dep_i = (struct polysa_dep *)malloc(sizeof(struct polysa_dep));
    dep_i = isl_basic_map_list_get_basic_map(deps, i);
    p_dep_i->isl_dep = isl_basic_map_copy(dep_i);

    isl_map *untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(isl_basic_map_copy(dep_i)));
    isl_basic_map *bmap_dep_i = isl_basic_map_from_map(untagged_dep_i);
    disvec = get_dep_dis_at_schedule(bmap_dep_i, schedule);
    /* The generated dependece distance vector contains the scalar dim, 
     * we will need to peel them off. */
    disvec = t2s_peel_off_scalar_dims_vec(disvec, schedule); 

//    // debug
//    isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//    p = isl_printer_print_basic_map(p, p_dep_i->isl_dep);
//    printf("\n");
//    p = isl_printer_print_vec(p, disvec);
//    printf("\n");
//    isl_printer_free(p);
//    // debug
    isl_basic_map_free(bmap_dep_i);

    isl_space *space = isl_basic_map_get_space(dep_i);
    isl_space *src_space = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
    isl_space *dest_space = isl_space_unwrap(isl_space_range(space));
    isl_id *src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
    isl_id *dest_id = isl_space_get_tuple_id(dest_space, isl_dim_out);
    isl_space_free(src_space);
    isl_space_free(dest_space);

    untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(dep_i));
    isl_set *src_domain = isl_map_domain(isl_map_copy(untagged_dep_i));
    isl_set *dest_domain = isl_map_range(untagged_dep_i);

    isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(band);
    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(isl_set_copy(src_domain)));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(isl_set_copy(dest_domain)));

    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));    
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));

    /* Add the tuple name */
    p_dep_i->src_sched_domain = isl_set_set_tuple_name(p_dep_i->src_sched_domain, isl_set_get_tuple_name(src_domain));
    p_dep_i->dest_sched_domain = isl_set_set_tuple_name(p_dep_i->dest_sched_domain, isl_set_get_tuple_name(dest_domain));
    isl_set_free(src_domain);
    isl_set_free(dest_domain);

    p_dep_i->src = src_id;
    p_dep_i->dest = dest_id;
    p_dep_i->disvec = disvec;
    p_dep_i->type = POLYSA_DEP_RAW;

    data->deps[i] = p_dep_i;
  }
  isl_basic_map_list_free(deps);

  /* Add RAR deps. */
  deps = isl_union_map_get_basic_map_list(dep_rar);
  ndeps = isl_union_map_n_basic_map(dep_rar);
  data->deps = (struct polysa_dep **)realloc(data->deps, (data->ndeps + ndeps) * sizeof(struct polysa_dep *));

  for (int i = 0; i < ndeps; i++) {
    p_dep_i = (struct polysa_dep *)malloc(sizeof(struct polysa_dep));
    dep_i = isl_basic_map_list_get_basic_map(deps, i);
    p_dep_i->isl_dep = isl_basic_map_copy(dep_i);

    isl_map *untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(isl_basic_map_copy(dep_i)));
    isl_basic_map *bmap_dep_i = isl_basic_map_from_map(untagged_dep_i);
    disvec = get_dep_dis_at_schedule(bmap_dep_i, schedule);
    /* The generated dependece distance vector contains the scalar dim, 
     * we will need to peel them off. */
    disvec = t2s_peel_off_scalar_dims_vec(disvec, schedule); 
    
    isl_basic_map_free(bmap_dep_i);

    isl_space *space = isl_basic_map_get_space(dep_i);
    isl_space *src_space = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
    isl_space *dest_space = isl_space_unwrap(isl_space_range(space));
    isl_id *src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
    isl_id *dest_id = isl_space_get_tuple_id(dest_space, isl_dim_out);
    isl_space_free(src_space);
    isl_space_free(dest_space);

    untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(dep_i));
    isl_set *src_domain = isl_map_domain(isl_map_copy(untagged_dep_i));
    isl_set *dest_domain = isl_map_range(untagged_dep_i);

//    // debug
//    isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//    p = isl_printer_print_set(p, src_domain);
//    printf("\n");
//    // debug

    isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(band);
//    // debug
//    p = isl_printer_print_union_map(p, sched);
//    printf("\n");
//    // debug

    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(isl_set_copy(src_domain)));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(isl_set_copy(dest_domain)));
    
    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));

//    // debug
//    p = isl_printer_print_set(p, p_dep_i->src_sched_domain);
//    printf("\n");
//    // debug

    /* Add the tuple name */
    p_dep_i->src_sched_domain = isl_set_set_tuple_name(p_dep_i->src_sched_domain, isl_set_get_tuple_name(src_domain));
    p_dep_i->dest_sched_domain = isl_set_set_tuple_name(p_dep_i->dest_sched_domain, isl_set_get_tuple_name(dest_domain));
    isl_set_free(src_domain);
    isl_set_free(dest_domain);
    
    p_dep_i->src = src_id;
    p_dep_i->dest = dest_id;
    p_dep_i->disvec = disvec;
    p_dep_i->type = POLYSA_DEP_RAR;

    data->deps[i + data->ndeps] = p_dep_i;
  }
  data->ndeps += ndeps;
  isl_basic_map_list_free(deps);

  /* Add WAW deps. */
  deps = isl_union_map_get_basic_map_list(dep_waw);
  ndeps = isl_union_map_n_basic_map(dep_waw);
  data->deps = (struct polysa_dep **)realloc(data->deps, (data->ndeps + ndeps) * sizeof(struct polysa_dep *));

  for (int i = 0; i < ndeps; i++) {
    p_dep_i = (struct polysa_dep *)malloc(sizeof(struct polysa_dep));
    dep_i = isl_basic_map_list_get_basic_map(deps, i);
    p_dep_i->isl_dep = isl_basic_map_copy(dep_i);

    isl_map *untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(isl_basic_map_copy(dep_i)));
    isl_basic_map *bmap_dep_i = isl_basic_map_from_map(untagged_dep_i);
    disvec = get_dep_dis_at_schedule(bmap_dep_i, schedule);
    /* The generated dependece distance vector contains the scalar dim, 
     * we will need to peel them off. */
    disvec = t2s_peel_off_scalar_dims_vec(disvec, schedule); 
    
    isl_basic_map_free(bmap_dep_i);

    isl_space *space = isl_basic_map_get_space(dep_i);
    isl_space *src_space = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
    isl_space *dest_space = isl_space_unwrap(isl_space_range(space));
    isl_id *src_id = isl_space_get_tuple_id(src_space, isl_dim_out);
    isl_id *dest_id = isl_space_get_tuple_id(dest_space, isl_dim_out);
    isl_space_free(src_space);
    isl_space_free(dest_space);

    untagged_dep_i = isl_map_factor_domain(isl_map_from_basic_map(dep_i));
    isl_set *src_domain = isl_map_domain(isl_map_copy(untagged_dep_i));
    isl_set *dest_domain = isl_map_range(untagged_dep_i);

    isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(band);
    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(isl_set_copy(src_domain)));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(isl_set_copy(dest_domain)));
    
    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));

    /* Add the tuple name */
    p_dep_i->src_sched_domain = isl_set_set_tuple_name(p_dep_i->src_sched_domain, isl_set_get_tuple_name(src_domain));
    p_dep_i->dest_sched_domain = isl_set_set_tuple_name(p_dep_i->dest_sched_domain, isl_set_get_tuple_name(dest_domain));
    isl_set_free(src_domain);
    isl_set_free(dest_domain);

    p_dep_i->src = src_id;
    p_dep_i->dest = dest_id;
    p_dep_i->disvec = disvec;
    p_dep_i->type = POLYSA_DEP_WAW;

    data->deps[i + data->ndeps] = p_dep_i; 
  }
  data->ndeps += ndeps;
  isl_basic_map_list_free(deps);

  isl_schedule_node_free(band);

  return schedule;
}

static __isl_null struct t2s_URE *t2s_URE_free(__isl_take struct t2s_URE *u) {
  if (!u)
    return NULL;

  free(u->name);
  free(u->text);
  free(u);

  return NULL;
}

__isl_null struct t2s_data *t2s_data_free(__isl_take struct t2s_data *d) {
  if (!d)
    return NULL;

  isl_set_free(d->anchor_domain);
  isl_union_set_free(d->stmt_domain);
  isl_union_set_free(d->stmt_sim_domain);
  for (int i = 0; i < d->t2s_stmt_num; i++) {
    free(d->t2s_stmt_text[i]);
  }
  free(d->t2s_stmt_text);

  for (int i = 0; i < d->URE_num; i++) {
    t2s_URE_free(d->URE[i]);
  }
  free(d->URE);

  for (int i = 0; i < d->iter_num; i++) {
    polysa_iter_free(d->iter[i]);
  }
  free(d->iter);

  isl_printer_free(d->p);
  for (int i = 0; i < d->ndeps; i++) {
    polysa_dep_free(d->deps[i]);    
  }
  free(d->deps);
  t2s_stmt_data_free(d->stmt_data); 

  isl_id_to_id_free(d->ref2func);
  for (int i = 0; i < d->n_array; i++) {
    struct t2s_array_info *array = &d->array[i];
    for (int j = 0; j < array->n_group; j++) {
      t2s_array_ref_group_free(array->groups[j]);
    }
    free(array->groups);
  }
  free(d->array);
  t2s_group_data_free(d->group_data);

  isl_id_list_free(d->func_ids);

  free(d);

  return NULL;
}

/* Generate T2S headers. */
static isl_stat gen_t2s_headers(struct t2s_data *data)
{
  isl_printer *p = data->p;
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "#include \"Halide.h\"");
  p = isl_printer_end_line(p);
  p = isl_printer_print_str(p, "#include <iostream>");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "using namespace Halide;");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "using namespace std;\n");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  return isl_stat_ok;
}

/* Generate T2S inputs. */
static isl_stat gen_t2s_inputs(struct t2s_data *data)
{
  isl_printer *p = data->p;
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Inputs (Fill in manually)");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);
}

/* Generate T2S variable declarations. */
static isl_stat gen_t2s_vars(struct t2s_data *data)
{
  isl_printer *p = data->p;
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Variable declarations");
  p = isl_printer_end_line(p);
  p = isl_printer_print_str(p, "Var ");
  for (int i = 0; i < data->iter_num; i++) {
    char iter_str[100];
    sprintf(iter_str, "c%d", i);
    if (i > 0) {
      p = isl_printer_print_str(p, ", ");
    }
    p = isl_printer_print_str(p, iter_str);
  }
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  return isl_stat_ok;
}

/* Fill up the group arrays with singleton groups, i.e., one group
 * per reference, initializing the array, access, write, n_ref and refs fields.
 * In particular the access field is initialized to the scheduled access relation 
 * of the array references.
 *
 * Return the number of elements initialized, i.e., the number of 
 * active references in the current kernel.
 */
static int t2s_populate_array_references(struct t2s_array_info *info, struct t2s_array_ref_group **groups, struct t2s_data *data) 
{
  isl_ctx *ctx = data->ctx;
  int n = 0;

  for (int i = 0; i < info->array->n_ref; i++) {
    isl_union_map *umap;
    isl_map *map;
    struct t2s_array_ref_group *group;
    struct gpu_stmt_access *access = info->array->refs[i];

    map = isl_map_copy(access->access);
    umap = isl_union_map_from_map(map);
    umap = isl_union_map_apply_domain(umap,
        isl_union_map_copy(data->group_data->full_sched));

    if (isl_union_map_is_empty(umap)) {
      isl_union_map_free(umap);
      continue;
    }

    map = isl_map_from_union_map(umap);
    map = isl_map_detect_equalities(map);

    group = isl_calloc_type(ctx, struct t2s_array_ref_group);
    if (!group) {
      isl_map_free(map);
      return -1;
    }

    group->t2s_array = info;
    group->array = info->array;
    group->access = map;
    group->write = access->write;
    group->exact_write = access->exact_write;
    group->refs = &info->array->refs[i];
    group->n_ref = 1;

    groups[n++] = group;
  }

  return n;
}

static struct t2s_array_ref_group *join_groups(
  struct t2s_array_ref_group *group1,
  struct t2s_array_ref_group *group2)
{
  isl_ctx *ctx;
  struct t2s_array_ref_group *group;

  if (!group1 || !group2) 
    return NULL;

  ctx = isl_map_get_ctx(group1->access);
  group = isl_calloc_type(ctx, struct t2s_array_ref_group);
  if (!group)
    return NULL;
  group->t2s_array = group1->t2s_array;
  group->array = group1->array;
  group->access = isl_map_union(isl_map_copy(group1->access),
      isl_map_copy(group2->access));
  group->write = group1->write || group2->write;
  group->exact_write = group1->exact_write && group2->exact_write;
  group->n_ref = group1->n_ref + group2->n_ref;
  group->refs = isl_alloc_array(ctx, struct gpu_stmt_access *,
      group->n_ref);
  if (!group->refs)
    return t2s_array_ref_group_free(group);
  for (int i = 0; i < group1->n_ref; i++)
    group->refs[i] = group1->refs[i];
  for (int i = 0; i < group2->n_ref; i++)
    group->refs[group1->n_ref + i] = group2->refs[i];

  return group;
}

/* Combine the given two groups into a single group and free
 * the original two groups. 
 */
static struct t2s_array_ref_group *join_groups_and_free(
  struct t2s_array_ref_group *group1,
  struct t2s_array_ref_group *group2)
{
  struct t2s_array_ref_group *group;

  group = join_groups(group1, group2);
  t2s_array_ref_group_free(group1);
  t2s_array_ref_group_free(group2);
  return group;
}

/* Group two writes if their live-in ranges overalp at the current iteration. */
static int t2s_group_writes(int n, struct t2s_array_ref_group **groups, 
    int (*overlap)(struct t2s_array_ref_group *group1, 
      struct t2s_array_ref_group *group2, void *user), struct t2s_data *data) {
  int i, j;
  int any_merge;

  for (i = 0; i < n; i += !any_merge) {
    any_merge = 0;
    for (j = n - 1; j > i; j--) {
      if (!overlap(groups[i], groups[j], data))
        continue;

      any_merge = 1;
      groups[i] = join_groups_and_free(groups[i], groups[j]);
      if (j != n - 1)
        groups[j] = groups[n - 1];
      groups[n - 1] = NULL;
      n--;

      if (!groups[i])
        return -1;      
    }
  }
  return n;
}

/* For each dependence, if the dependence distance are all zero by the members of the schedule band,
 * then, compute the live-range from the src to the dest of the dependence.
 * Otherwise, compute the live-range by not considering the dest of the dependence.
 */
static __isl_give isl_set *t2s_compute_dep_live_range(struct polysa_dep *d, struct t2s_data *data) {
  isl_basic_map *bmap;
  isl_basic_set *bset;
  isl_map *map;
  isl_set *set;
  isl_set *src_set;
  isl_set *dest_set;
  isl_map *lex_map;
  isl_union_map *sched;
  isl_set *live_range;

  bmap = isl_basic_map_copy(d->isl_dep);
  map = isl_map_factor_domain(isl_map_from_basic_map(bmap));
  set = isl_map_domain(map);
  sched = isl_union_map_copy(data->group_data->full_sched);
  sched = isl_union_map_intersect_domain(sched, isl_union_set_from_set(set));
  set = isl_map_range(isl_map_from_union_map(sched));
  lex_map = isl_map_lex_le(isl_set_get_space(set));
  for (int i = 0; i < data->iter_num; i++) {
    lex_map = isl_map_equate(lex_map, isl_dim_in, i, isl_dim_out, i);
  }

  src_set = isl_set_apply(set, lex_map);
  if (isl_vec_is_zero(d->disvec)) {
    bmap = isl_basic_map_copy(d->isl_dep);
    map = isl_map_factor_domain(isl_map_from_basic_map(bmap));
    set = isl_map_range(map);
    sched = isl_union_map_copy(data->group_data->full_sched);
    sched = isl_union_map_intersect_domain(sched, isl_union_set_from_set(set));
    set = isl_map_range(isl_map_from_union_map(sched));
    lex_map = isl_map_lex_gt(isl_set_get_space(set));
    for (int i = 0; i < data->iter_num; i++) {
      lex_map = isl_map_equate(lex_map, isl_dim_in, i, isl_dim_out, i);
    }
    dest_set = isl_set_apply(set, lex_map);
    live_range = isl_set_intersect(src_set, dest_set);
  } else {
    live_range = src_set;
  }

  return live_range;
}

static int accesses_overlap(struct polysa_dep *d1, struct polysa_dep *d2, int wr,
    struct t2s_data *data) 
{
  isl_set *live_range1;
  isl_set *live_range2;
  int r;

  live_range1 = t2s_compute_dep_live_range(d1, data);
  live_range2 = t2s_compute_dep_live_range(d2, data);
  if (isl_set_is_disjoint(live_range1, live_range2)) {
    r = 0;
  } else {
    r = 1;
  }

  isl_set_free(live_range1);
  isl_set_free(live_range2);

  return r;
}

/* If both accesses are write accesses with RAW dep (intermediate access),
 * we will check if:
 * 1) both writes are local to the permutable band, i.e., if they are 
 * assigned the same value by the members of the band.
 * 2) if the first condition holds, check if the live-ranges (RAW) of two accesses
 * are overlapped.
 * If both conditions hold, which means that these two write accesses 
 * need to be assigned different T2S function names to avoid overwriting 
 * the value of each other. We will return 1 and later group them into the 
 * same array reference group. 
 *
 * If both accesses are read accesses with RAR dep (intermediate access),
 * we will check if:
 * 1) both read are local to the permutable band, i.e., if they are 
 * assigned the same value by the members of the band.
 * 2) if the first condition holds, check if the live-ranges (RAR) of two accesses
 * are overlapped.
 * If both conditions hold, which means that these two read accesses 
 * need to be assigned different T2S function names to avoid overwriting
 * the value of each other. We will return 1 and later group them into the 
 * same array reference group.
 *
 * For live-out accesses (drain accesses), we will need to assign them different
 * function names if they are updated in the same iteration. Currently, we don't
 * handle this scenario as there is only one statement that involves the live-out
 * accesses for all of the test cases.
 */
static int accesses_overlap_wrap(struct t2s_array_ref_group *group1, 
    struct t2s_array_ref_group *group2, void *user)
{
  struct t2s_data *data = user;
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_map_get_ctx(group1->access), stdout);
//  p = isl_printer_print_map(p, group1->access);
//  printf("\n");
//  p = isl_printer_print_map(p, group2->access);
//  printf("\n");
//  isl_printer_free(p);
//  // debug

  for (int i = 0; i < group1->n_ref; i++) {
    for (int j = 0; j < group2->n_ref; j++) {
      struct gpu_stmt_access *ref1 = group1->refs[i];
      struct gpu_stmt_access *ref2 = group2->refs[j];

      if (ref1->write == 1 && ref2->write == 1) {
        for (int n = 0; n < data->ndeps; n++) {
          struct polysa_dep *dep1 = data->deps[n];
          if (dep1->type == POLYSA_DEP_RAW && dep1->src == ref1->ref_id) {
            for (int m = 0; m < data->ndeps; m++) {
              struct polysa_dep *dep2 = data->deps[m];
              if (dep2->type == POLYSA_DEP_RAW && dep2->src == ref2->ref_id) {
                /* Examine if two write accesses are overlapped. */
                return accesses_overlap(dep1, dep2, 0, data);
              }
            }
          }
        }
      } else if (ref1->read == 1 && ref2->read == 1) {
        for (int n = 0; n < data->ndeps; n++) {
          struct polysa_dep *dep1 = data->deps[n];
          if (dep1->type == POLYSA_DEP_RAR && dep1->src == ref1->ref_id) {
            for (int m = 0; m < data->ndeps; m++) {
              struct polysa_dep *dep2 = data->deps[m];
              if (dep2->type == POLYSA_DEP_RAR && dep2->src == ref2->ref_id) {
                /* Examine if two read accesses are overlapped. */
                return accesses_overlap(dep1, dep2, 1, data);
              }
            }
          }
        }       
      } else {
        return 0;
      }
    }
  }
  return 0;
}

static int t2s_group_overlapping_writes(int n, struct t2s_array_ref_group **groups, struct t2s_data *data) 
{
  return t2s_group_writes(n, groups, &accesses_overlap_wrap, data);
}

/* Set array->n_group and array->groups to n and groups.
 *
 * Additionally, set the "nr" field of each group.
 */
static void t2s_set_array_groups(struct t2s_array_info *array, int n, struct t2s_array_ref_group **groups) 
{
  int i;

  array->n_group = n;
  array->groups = groups;

  for (i = 0; i < n; i++) {
    groups[i]->nr = i;
  }
}

static void t2s_assign_array_group_func_id(struct t2s_data *data, struct t2s_array_info *array)
{
  int max_n_ref = 0;
  int write = 0;

  if (data->ref2func == NULL)
    data->ref2func = isl_id_to_id_alloc(data->ctx, 0);
 
  if (data->func_ids == NULL)
    data->func_ids = isl_id_list_alloc(data->ctx, 0);

  for (int i = 0; i < array->n_group; i++) {
    struct t2s_array_ref_group *group = array->groups[i];
    if (group->write == 1)
      write = 1;

    for (int r = 0; r < group->n_ref; r++) {
      struct gpu_stmt_access *ref = group->refs[r];
      char func_name[100];
      /* Fetch the array name */
      isl_map *access = isl_map_copy(ref->access);
      isl_id *ref_id = isl_id_copy(ref->ref_id);
      isl_space *space = isl_map_get_space(access);
      const char *array_name = isl_space_get_tuple_name(space, isl_dim_out);
      isl_space_free(space);
      isl_map_free(access);
      if (r == 0) {
        sprintf(func_name, "%s", array_name);
      } else {
        sprintf(func_name, "%s_%d", array_name, r);
      }
      isl_id *func_id = isl_id_alloc(data->ctx, func_name, NULL);
      data->ref2func = isl_id_to_id_set(data->ref2func, ref_id, func_id);
    }
    if (group->n_ref > max_n_ref)
      max_n_ref = group->n_ref;
  }

//  // debug
//  printf("%s\n", array->array->type);
//  // debug

  /* Insert the function declaraitons. */
  for (int i = 0; i < max_n_ref; i++) {
    char func_name[100];
    if (i == 0) 
      sprintf(func_name, "%s", array->array->name);
    else
      sprintf(func_name, "%s_%d", array->array->name, i);
    isl_id *func_id = isl_id_alloc(data->ctx, func_name, array->array);
    data->func_ids = isl_id_list_add(data->func_ids, func_id);

    if (write == 1) {
      /* Add the drain func. */
      char func_name[100];
      if (i == 0) 
        sprintf(func_name, "%s_drain", array->array->name);
      else
        sprintf(func_name, "%s_%d_drain", array->array->name, i);
      isl_id *func_id = isl_id_alloc(data->ctx, func_name, array->array);
      data->func_ids = isl_id_list_add(data->func_ids, func_id);
    }
  }
}

static isl_stat gen_t2s_func_ids(struct t2s_data *data, struct t2s_array_info *array) {
  /* Populate the array groups. */
  isl_ctx *ctx = data->ctx;
  struct t2s_array_ref_group **groups;
  groups = isl_calloc_array(ctx, struct t2s_array_ref_group *, array->array->n_ref);

  int n = t2s_populate_array_references(array, groups, data);

  /* Group overlapping writes. */
  n = t2s_group_overlapping_writes(n, groups, data);  

  /* Set the group information. */
  t2s_set_array_groups(array, n, groups); 

  /* Assign function names. */
  t2s_assign_array_group_func_id(data, array);
    
  return isl_stat_ok;
}


/* Generate function declarations. 
 * Assign a function name to each access reference.
 * First group references that access the same array together.
 * Connect all accesses in the same group together.
 * Inside each group, if the access is a write access (assocaited with
 * RAW), check if there is any other write access scheduled in-between this
 * write access and the read access that uses this data by RAW.
 * If so, break the edge between these two write accesses.
 * At last, compute the CCs of the graph, and assign a unique function name 
 * to each CC of the array group.
 *
 * Inside each group, if the access is a read access (associated with
 * RAR), check if there is any other read access scheduled in-between this 
 * read access and the read access that uses this data by RAR.
 * If so, break the edge between these two read accesses.
 * At last, compute the CCs of the graph, and assign a unique function name
 * to each CC of the array group.
 * 
 * Inside each group, if the access is a write access (associated with 
 * WAW), check if the write-out domain is empty. If not, generate a unique function 
 * name to this write access as the drain function.
 */
static isl_stat gen_t2s_funcs(__isl_keep isl_schedule *schedule, struct t2s_data *data)
{
  isl_ctx *ctx = data->ctx;
  isl_printer *p = data->p;
  struct t2s_group_data *group_data;
  struct gpu_prog *prog;
  isl_schedule_node *node;
  
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Function declarations");
  p = isl_printer_end_line(p);

  /* Initialization. */
  prog = gpu_prog_alloc(ctx, data->scop);
  group_data = isl_calloc_type(ctx, struct t2s_group_data);

  data->array = isl_calloc_array(ctx, struct t2s_array_info, prog->n_array);
  data->n_array = prog->n_array;
  for (int i = 0; i < prog->n_array; i++) {
    data->array[i].array = &prog->array[i];
  }
  node = isl_schedule_get_root(schedule);
  group_data->full_sched = isl_schedule_node_get_subtree_schedule_union_map(node);
  isl_schedule_node_free(node);
  data->group_data = group_data;

  for (int i = 0; i < data->n_array; i++) {
    gen_t2s_func_ids(data, &data->array[i]);
  }
//  // debug
//  isl_printer *p_debug = isl_printer_to_file(data->ctx, stdout);
//  p_debug = isl_printer_print_id_to_id(p_debug, data->ref2func);
//  printf("\n");
//  isl_printer_free(p_debug);
//  // debug

  data->group_data = t2s_group_data_free(group_data);

  /* Print the function decls. */
  for (int i = 0; i < isl_id_list_n_id(data->func_ids); i++) {
    isl_id *func_id = isl_id_list_get_id(data->func_ids, i);
    struct gpu_array_info *array = isl_id_get_user(func_id);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#define FUNC_S");
    p = isl_printer_print_int(p, i);
    p = isl_printer_print_str(p, " type_of<");
    p = isl_printer_print_str(p, array->type);
    p = isl_printer_print_str(p, ">(), {");
    for (int j = 0; j < data->iter_num; j++) {
      if (j > 0)
        p = isl_printer_print_str(p, ", ");
      p = isl_printer_print_str(p, "c");
      p = isl_printer_print_int(p, j);
    }
    p = isl_printer_print_str(p, "}, Place::Host");
    p = isl_printer_end_line(p);
    isl_id_free(func_id);
  }
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "Func ");
  for (int i = 0; i < isl_id_list_n_id(data->func_ids); i++) {
    isl_id *func_id = isl_id_list_get_id(data->func_ids, i);
    if (i > 0) {
      p = isl_printer_print_str(p, ", ");
    }
    p = isl_printer_print_str(p, isl_id_get_name(func_id));
    p = isl_printer_print_str(p, "(FUNC_S");
    p = isl_printer_print_int(p, i);
    p = isl_printer_print_str(p, ")");
    isl_id_free(func_id);
  }
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  gpu_prog_free(prog);
  return isl_stat_ok;
}

/* Generate T2S space-time transformation. */
static isl_stat gen_t2s_space_time(struct t2s_data *data)
{
  struct t2s_URE *d_URE;

  isl_printer *p = data->p;
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Space-time transformation");
  p = isl_printer_end_line(p);

  /* Define time and space loop variables. */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "Var ");
  int is_var_first = 1;
  for (int i = 0; i < data->prog->time_w; i++) {    
    if (!is_var_first) {
      p = isl_printer_print_str(p, ", ");
    }
    p = isl_printer_print_str(p, "tloop");
    p = isl_printer_print_int(p, i);
    if (is_var_first) {
      is_var_first = 0;
    }
  }
  for (int i = 0; i < data->prog->space_w; i++) {    
    if (!is_var_first) {
      p = isl_printer_print_str(p, ", ");
    }
    p = isl_printer_print_str(p, "sloop");
    p = isl_printer_print_int(p, i);
    if (is_var_first) {
      is_var_first = 0;
    }
  }
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  for (int i = 0; i < data->URE_num; i++) {
    if (data->URE[i]->d == 1) {
      d_URE = data->URE[i];
      if (d_URE->update_level == -1)
        break;
    }
  }
  assert(d_URE->update_level == -1);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, d_URE->name);
  p = isl_printer_print_str(p, ".");
  p = isl_printer_print_str(p, "merge_defs(");
  p = isl_printer_print_str(p, "{");
  int is_first = 1;
  for (int i = 0; i < data->URE_num; i++) {
    struct t2s_URE *URE = data->URE[i];
    if (URE->update_level >= 0) {
      if (!is_first) {
        p = isl_printer_print_str(p, ", ");
      }
      p = isl_printer_print_str(p, URE->name);
      if (is_first)
        is_first = 0;
    }
  }
  p = isl_printer_print_str(p, "}, {");
  is_first = 1;
  for (int i = 0; i < data->URE_num; i++) {
    struct t2s_URE *URE = data->URE[i];
    if (URE->update_level == -1 && URE->d == 0) {
      if (!is_first) {
        p = isl_printer_print_str(p, ", ");
      }
      p = isl_printer_print_str(p, URE->name);
      if (is_first)
        is_first = 0;
    }
  }
  p = isl_printer_print_str(p, "}");
  p = isl_printer_print_str(p, ")");
  p = isl_printer_end_line(p);

  p = isl_printer_indent(p, strlen(d_URE->name));
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, ".reorder_inward(");
  is_first = 1;
  for (int i = 0; i < data->iter_num; i++) {
    if (!is_first) 
      p = isl_printer_print_str(p, ", ");
    char iter_name[100];
    sprintf(iter_name, "c%d", i);
    p = isl_printer_print_str(p, iter_name);
    if (is_first)
      is_first = 0;
  }
  p = isl_printer_print_str(p, ")");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, ".space_time_transform(");
  // TODO
  int indent = strlen(".space_time_transform(");
  p = isl_printer_indent(p, indent);
  p = isl_printer_print_str(p, "{");
  for (int i = 0; i < data->iter_num; i++) {
    if (i > 0)
      p = isl_printer_print_str(p, ", ");
    char iter_name[100];
    sprintf(iter_name, "c%d", i);
    p = isl_printer_print_str(p, iter_name);
  }
  p = isl_printer_print_str(p, "},");
  p = isl_printer_end_line(p);
  /* Space and time loops. */
  isl_printer *p_str = isl_printer_to_str(data->ctx);
  p_str = isl_printer_print_str(p_str, "{");
  for (int i = 0; i < data->prog->time_w; i++) {
    if (i > 0)
      p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, "tloop");
    p_str = isl_printer_print_int(p_str, i);
  }
  p_str = isl_printer_print_str(p_str, "}");
  char *tloop_list = isl_printer_get_str(p_str);
  isl_printer_free(p_str);
  
  p_str = isl_printer_to_str(data->ctx);
  p_str = isl_printer_print_str(p_str, "{");
  for (int i = 0; i < data->prog->space_w; i++) {
    if (i > 0)
      p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, "sloop");
    p_str = isl_printer_print_int(p_str, i);
  }
  p_str = isl_printer_print_str(p_str, "}");
  char *sloop_list = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  if (data->prog->type == POLYSA_SA_TYPE_ASYNC) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, sloop_list);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, tloop_list);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  } else if (data->prog->type == POLYSA_SA_TYPE_SYNC) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, tloop_list);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, sloop_list);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  }
  free(tloop_list);
  free(sloop_list);
  /* Transform matrix. */
  for (int i = 0; i < data->iter_num; i++) {
    p = isl_printer_start_line(p);
    if (i == 0) {
      p = isl_printer_print_str(p, "{");
      p = isl_printer_indent(p, 1);
    }
    for (int j = 0; j < data->iter_num; j++) {
      if (j > 0)
        p = isl_printer_print_str(p, ", ");
      if (i == j)
        p = isl_printer_print_int(p, 1);
      else 
        p = isl_printer_print_int(p, 0);
    }

    if (i == data->iter_num - 1) {
      p = isl_printer_print_str(p, "}");
      p = isl_printer_indent(p, -1);
    }
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  }
  /* Reverse transform matrix. */
  for (int i = 0; i < data->iter_num; i++) {
    p = isl_printer_start_line(p);
    if (i == 0) {
      p = isl_printer_print_str(p, "{");
      p = isl_printer_indent(p, 1);
    }
    for (int j = 0; j < data->iter_num; j++) {
      if (j > 0)
        p = isl_printer_print_str(p, ", ");
      if (i == j)
        p = isl_printer_print_int(p, 1);
      else 
        p = isl_printer_print_int(p, 0);
    }

    if (i == data->iter_num - 1) {
      p = isl_printer_print_str(p, "}");
      p = isl_printer_indent(p, -1);
    }
    if (i == data->iter_num - 1)
      p = isl_printer_print_str(p, ")");
    else
      p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_indent(p, -indent);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, ".domain(");
  int indent_tmp = strlen(".domain(");
  p = isl_printer_indent(p, indent_tmp);
  for (int i = 0; i < data->iter_num; i++) {
    if (i > 0)
      p = isl_printer_start_line(p);
    struct polysa_iter *iter = data->iter[i];    
    p = isl_printer_print_str(p, iter->name);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
    p = isl_printer_print_aff(p, iter->lb);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_aff(p, iter->ub);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_int(p, iter->stride);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  }
  /* Add time and space loops. */
  for (int i = 0; i < data->iter_num; i++) {
    p = isl_printer_start_line(p);
    struct polysa_iter *iter = data->iter[i];    
    p = isl_printer_print_str(p, iter->ts_name);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
    p = isl_printer_print_aff(p, iter->lb);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_aff(p, iter->ub);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_int(p, iter->stride);
    if (i == data->iter_num - 1)
      p = isl_printer_print_str(p, ");");
    else
      p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
  }

  p = isl_printer_indent(p, -indent_tmp);
  p = isl_printer_set_output_format(p, ISL_FORMAT_ISL);

  p = isl_printer_indent(p, -strlen(d_URE->name));
  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  return isl_stat_ok;
}

static __isl_give struct t2s_data *t2s_data_init(__isl_take struct t2s_data *d) {
  d->anchor_domain = NULL;
  d->stmt_domain = NULL;
  d->stmt_sim_domain = NULL;
  d->URE = NULL;
  d->URE_num = 0;
  d->t2s_stmt_num = 0;
  d->t2s_stmt_text = NULL;
  d->iter_num = 0;
  d->iter = NULL;
  d->scop = NULL;
  d->p = NULL;
  d->ctx = NULL;
  d->deps = NULL;
  d->ndeps = 0;
  d->ref2func = NULL;
//  d->ref2dfunc = NULL;
  d->func_ids = NULL;
  d->stmt_data = NULL;
  d->n_array = 0;
  d->array = NULL;
  d->group_data = NULL;
  d->prog = NULL;
  d->schedule = NULL;
}

static isl_stat extract_anchor_domain(__isl_keep isl_schedule *schedule, struct t2s_data *data) {
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_union_set *domain = isl_schedule_get_domain(schedule);
  isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(root);
  isl_schedule_node_free(root);

  isl_union_set *anchor_domain = isl_union_set_apply(domain, sched);
  
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_set_get_ctx(anchor_domain), stdout);
//  p = isl_printer_print_union_set(p, anchor_domain);
//  printf("\n");
//  // debug

  data->anchor_domain = isl_set_from_union_set(anchor_domain);
  return isl_stat_ok;
}

/* Generate T2S code from schedule. */
static isl_stat print_t2s_with_schedule(
    __isl_keep struct polysa_kernel *prog,
    __isl_keep struct ppcg_scop *scop)
{
  struct t2s_data *data;
  isl_union_set *stmt_domains;
  isl_union_map *stmt_schedules;
  isl_schedule_node *node;
  isl_union_set *stmt_trans_domains;
  isl_space *stmt_space;
  isl_ctx *ctx;
  isl_schedule *schedule;
  int t2s_tile_second_phase;

  schedule = prog->schedule;
  ctx = isl_schedule_get_ctx(schedule);
  data = isl_calloc_type(ctx, struct t2s_data);
  data = t2s_data_init(data);
  t2s_tile_second_phase = (scop->options->t2s_tile && scop->options->t2s_tile_phase == 1);

  data->ctx = ctx;
  data->scop = scop;
  data->prog = prog;
  data->schedule = schedule;
  FILE *t2s_fp = fopen("t2s.cpp", "w");
  data->p = isl_printer_to_file(ctx, t2s_fp); 

  /* Print out the headers. */
  gen_t2s_headers(data);
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_print_str(data->p, "int main(void) {");
  data->p = isl_printer_end_line(data->p);

  /* Calcualte the iterator num. */  
  data->iter_num = 0;
  isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth, &data->iter_num);

  /* Update the deps. */
  data->ndeps = 0;
  data->deps = NULL;
  schedule = extract_deps(schedule, data);

  /* Calculate the anchor domain. 
   * Allocate a empty set then unionize it with the scheduling domain of each statement. 
   */
  data->anchor_domain = NULL;
  extract_anchor_domain(schedule, data);

  /* Generate the iterator meta data. */
  extract_iters(schedule, data);

  /* Calculate the simplified domain (in scheduling dims) for each statement. */
  data->stmt_domain = NULL;
  data->stmt_sim_domain = NULL;
  extract_stmt_domain(schedule, data);

  /* Generate input declarations. */
  gen_t2s_inputs(data);

  /* Generate variable declarations. */
  gen_t2s_vars(data);

  /* Generate function declarations. */
  gen_t2s_funcs(schedule, data);

  /* Generate the T2S statements .*/ 
  data->t2s_stmt_num = 0;
  data->t2s_stmt_text = NULL;
  schedule = gen_stmt_text_wrap(schedule, data);
  
  /* Generate time-space transformation. */
  if (!t2s_tile_second_phase) {
    gen_t2s_space_time(data);
  } else {
    data->p = isl_printer_start_line(data->p);
    data->p = isl_printer_print_str(data->p, "// Space-time transformation (Fill in manually)");
    data->p = isl_printer_end_line(data->p);
    data->p = isl_printer_start_line(data->p);
    data->p = isl_printer_end_line(data->p);
  }

  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_print_str(data->p, "// PE optimization (Fill in manually)");
  data->p = isl_printer_end_line(data->p);
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_end_line(data->p);

  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_print_str(data->p, "// CPU verification (Fill in manually)");
  data->p = isl_printer_end_line(data->p);
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_end_line(data->p);

  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_print_str(data->p, "}");
  data->p = isl_printer_end_line(data->p);

  fclose(t2s_fp);
  t2s_data_free(data);

  prog->schedule = schedule;
  return isl_stat_ok;
}

static isl_bool no_band_node_as_descendant(__isl_keep isl_schedule_node *node, void *user){
  enum isl_schedule_node_type node_type = isl_schedule_node_get_type(node);
  if (node_type == isl_schedule_node_band) {
    return isl_bool_false;
  } else {
    return isl_bool_true;
  }
}

/* No band node is allowed after the sequence or set node. */
static isl_bool t2s_legal_at_node(__isl_keep isl_schedule_node *node, void *user) {
  enum isl_schedule_node_type node_type = isl_schedule_node_get_type(node);
  if (node_type == isl_schedule_node_sequence || node_type == isl_schedule_node_set) {
    int n_node_has_band = 0;
    for (int n = 0; n < isl_schedule_node_n_children(node); n++) {
      node = isl_schedule_node_child(node, n);
      isl_bool no_band = isl_schedule_node_every_descendant(node, 
          &no_band_node_as_descendant, NULL);
      if (!no_band) 
        n_node_has_band++;  
    }
    if (n_node_has_band > 2) {
      return isl_bool_false;
    } else {
      return isl_bool_true;
    }
  } else {
    return isl_bool_true;
  }
}

/* Check if there is only nested permutable band in the program.
 */
static isl_bool t2s_legality_check(__isl_keep isl_schedule *schedule) {
  isl_schedule_node *root = isl_schedule_get_root(schedule);
  isl_bool is_legal = isl_schedule_node_every_descendant(root,
      &t2s_legal_at_node, NULL);
  isl_schedule_node_free(root);

  return is_legal;
}

/* Generate CPU code for "scop" and print it to "p".
 *
 * First obtain a schedule for "scop" and then print code for "scop"
 * using that schedule.
 *
 * To generate T2S code from the tiled design, there are two phases.
 * In the first phase, a tiled CPU program is generated w/o T2S program.
 * In the second phase, the tiled CPU program is taken in and the T2S program
 * with tiled UREs are generated.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct ppcg_scop *scop, struct ppcg_options *options)
{
	isl_schedule *schedule;
  int t2s_tile_second_phase = (options->t2s_tile && options->t2s_tile_phase == 1);
  int t2s_tile_first_phase = (options->t2s_tile && options->t2s_tile_phase == 0);

  if (t2s_tile_second_phase) {
    /* In the second phase, the reschedule is disabled so that the 
     * original schedule from the program is used. */
    options->reschedule = 0;
  }
	schedule = t2s_get_schedule(scop, options);

//  // debug
//  isl_printer *p_debug = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
//  p_debug = isl_printer_set_yaml_style(p_debug, ISL_YAML_STYLE_BLOCK);
//  p_debug = isl_printer_print_schedule(p_debug, schedule);
//  printf("\n");
//  p_debug = isl_printer_print_union_map(p_debug, scop->tagged_dep_flow);
//  printf("\n");
//  p_debug = isl_printer_print_union_map(p_debug, scop->tagged_dep_waw);
//  printf("\n");
//  p_debug = isl_printer_print_union_map(p_debug, scop->tagged_dep_rar);
//  printf("\n");
//  isl_printer_free(p_debug);
//  // debug
   
  if (!t2s_tile_second_phase) {
    /*  Check if the program is legal to be mapped to systolic array. */
    isl_bool is_legal = sa_legality_check(schedule, scop);
    if (is_legal != isl_bool_true) {
      printf("[PolySA] Illegal to be transformed to systolic array.\n");
    }
    
    if (is_legal) {
      /* Generate systolic arrays using space-time mapping. */
      isl_size num_sa = 0;
      struct polysa_kernel **sa_candidates = sa_space_time_transform(schedule, scop, &num_sa);
      if (num_sa > 0) {
        printf("[PolySA] %d systolic arrays generated.\n", num_sa);
      }
  
      // TODO: All the SA candidates keep the same schedule tree. We need to duplicate them to 
      // seperate the transformation performed on each array.
  
      /* Pick up one systolic array to proceed based on heuristics. */
      struct polysa_kernel *sa_opt = sa_candidates_smart_pick(sa_candidates, num_sa);
  
      if (t2s_tile_first_phase) {
        bool opt_en[3] = {1, 1, 1};
        char *opt_mode[3] = {"auto", "auto", "auto"};
        /* Apply PE optimization. */
        sa_pe_optimize(sa_opt, opt_en, opt_mode);
      }
  
  //    // debug
  //    // isl_printer *p = isl_printer_to_file(isl_schedule_get_ctx(sa_opt->schedule), stdout);
  //    p_debug = isl_printer_print_schedule(p_debug, sa_opt->schedule);
  //    printf("\n");
  //    // debug
 
      if (!t2s_tile_first_phase) {
        /* Generate T2S program. */
        isl_bool is_t2s_legal = t2s_legality_check(sa_opt->schedule);
        if (is_t2s_legal) {
          print_t2s_with_schedule(sa_opt, scop);
        } else {
          printf("[PolySA] Illegal to be transformed to T2S program.\n");
        }
      }

      schedule = isl_schedule_copy(sa_opt->schedule);  
      polysa_kernel_free(sa_opt);
    }
  } else {
    struct polysa_kernel *sa = polysa_kernel_from_schedule(schedule);
    sa->scop = scop;
    // TODO: sa->type
    // TODO: sa->array_dim
    // TODO: sa->array_part_w
    // TODO: sa->space_w
    // TODO: sa->time_w

    print_t2s_with_schedule(sa, scop);
    schedule = isl_schedule_copy(sa->schedule);
    polysa_kernel_free(sa);
  }

  /* Generate the transformed CPU program. */
	return print_cpu_with_schedule(p, scop, schedule, options);
}

/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *print_polysa_t2s_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct ppcg_options *options = user;

	return generate(p, scop, options);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CPU code and write the results to a file
 * called "output".
 */
int generate_polysa_t2s(isl_ctx *ctx, struct ppcg_options *options, 
    const char *input, const char *output)
{
  FILE *output_file;
  int r;

  /* Return the handle of the output file. */
  output_file = get_output_file(input, output);
  if (!output_file)
    return -1;

  /* Extract each scop from the program and call the callback 
   * function to process it. */
  r = ppcg_transform(ctx, input, output_file, options,
          &print_polysa_t2s_wrap, options);

  fclose(output_file);

  return r;
}

