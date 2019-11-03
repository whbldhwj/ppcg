#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <pet.h>
#include <pet/expr.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "print.h"
#include "schedule.h"
#include "util.h"
#include "polysa_cpu.h"

struct t2s_stmt {
  char *content;
};

struct polysa_stmt {
  struct ppcg_stmt *stmt;
  
  /* T2S */
  struct t2s_stmt *t_stmt;
};

/* Representation of a statement inside a generated AST.
 *
 * "stmt" refers to the original statement.
 * "ref2expr" maps the reference identifier of each access in
 * the statement to an AST expression that should be printed
 * at the place of the access.
 */
struct ppcg_stmt {
	struct pet_stmt *stmt;

	isl_id_to_ast_expr *ref2expr;
};

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

static void polysa_stmt_free(void *user)
{
  struct polysa_stmt *p_stmt = user;
  
  if (!p_stmt)
    return;

  ppcg_stmt_free(p_stmt->stmt);
  t2s_stmt_free(p_stmt->t_stmt);
  free(p_stmt);
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

static struct pet_stmt *find_stmt_wrap(struct ppcg_scop *scop, __isl_keep isl_id *id)
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

  if (i == scop->pet->n_stmt)
    return NULL;
}

/* Find the element in scop->stmts that has the given "id".
 */
static struct pet_stmt *find_stmt(struct ppcg_scop *scop, __isl_keep isl_id *id)
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
  struct polysa_stmt *p_stmt; // n
	isl_id *id;
  int is_orig;
  int is_t2s;

	id = isl_ast_node_get_annotation(node);
  p_stmt = isl_id_get_user(id);
	isl_id_free(id);

  is_orig = p_stmt->stmt == NULL ? 0: 1;
  is_t2s = p_stmt->t_stmt == NULL ? 0 : 1;

  if (is_orig) {
    p = pet_stmt_print_body(p_stmt->stmt->stmt, p, p_stmt->stmt->ref2expr);
//    // debug
//    isl_printer *pstr = isl_printer_to_str(isl_printer_get_ctx(p));
//	  pstr = isl_printer_set_output_format(pstr, ISL_FORMAT_C);
//    pstr = pet_stmt_print_body(p_stmt->stmt->stmt, pstr, p_stmt->stmt->ref2expr);
//    printf("%s\n", isl_printer_get_str(pstr));
//    FILE *fp = fopen("tmptmp", "w");
//    isl_printer *pp = isl_printer_to_file(isl_printer_get_ctx(p), fp);
//    pp = isl_printer_print_id_to_ast_expr(pp, p_stmt->stmt->ref2expr);
//    printf("\n");
//    pp = pet_stmt_print_body(p_stmt->stmt->stmt, pp, p_stmt->stmt->ref2expr);
//    printf("\n");
//    fclose(fp);
//    // debug
  }
  else if (is_t2s) {
    /* Print T2S stmt. */
    isl_printer **t2s_p = user;
    *t2s_p = isl_printer_start_line(*t2s_p);
    *t2s_p = isl_printer_print_str(*t2s_p, p_stmt->t_stmt->content);
    *t2s_p = isl_printer_end_line(*t2s_p);
  }

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
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_id_get_ctx(id), stdout);
//  p = isl_printer_print_multi_pw_aff(p, index);
//  printf("\n");
//  p = isl_printer_print_pw_multi_aff(p, iterator_map);
//  printf("\n");
//  isl_multi_pw_aff *new = isl_multi_pw_aff_pullback_pw_multi_aff(isl_multi_pw_aff_copy(index), isl_pw_multi_aff_copy(iterator_map));
//  p = isl_printer_print_multi_pw_aff(p, new);
//  printf("\n");
//  // debug
	return isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);
}

static int is_id_t2s(__isl_keep isl_id *id)
{
  const char *name;

  name = isl_id_get_name(id);
  if (!name)
    return 0;
  else if (strncmp(name, "t2s_stmt", 8))
    return 0;
  return 1;
}

static __isl_give isl_ast_node *create_t2s_leaf(
    char *content, __isl_take isl_ast_node *node, __isl_keep isl_ast_build *build)
{
  isl_id *id;
  struct polysa_stmt *p_stmt;
  struct t2s_stmt *t_stmt;

  p_stmt = isl_calloc_type(isl_ast_node_get_ctx(node), struct polysa_stmt);
  t_stmt = isl_calloc_type(isl_ast_node_get_ctx(node), struct t2s_stmt);
  if (!p_stmt)
    return isl_ast_node_free(node);
  if (!t_stmt)
    return isl_ast_node_free(node);
  
  p_stmt->stmt = NULL;
  p_stmt->t_stmt = t_stmt;
  p_stmt->t_stmt->content = content;
  id = isl_id_alloc(isl_ast_node_get_ctx(node), "t2s", p_stmt);
  id = isl_id_set_free_user(id, &polysa_stmt_free);
  if (!id)
    polysa_stmt_free(p_stmt);

  return isl_ast_node_set_annotation(node, id);
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
  struct polysa_stmt *p_stmt; // n
  struct t2s_stmt *t_stmt; // n
  void *p;
  const char *name;
  int is_t2s;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(expr);
	isl_ast_expr_free(arg);

  name = isl_id_get_name(id);
  p = isl_id_get_user(id); // n

  is_t2s = is_id_t2s(id);
  if (is_t2s) {
    isl_id_free(id);
    char *content = p;
    return create_t2s_leaf(p, node, build);
  }

	ctx = isl_ast_node_get_ctx(node);
	stmt = isl_calloc_type(ctx, struct ppcg_stmt);
	stmt->stmt = find_stmt_wrap(scop, id);
  isl_id_free(id);

  /* Normal user statement. */
  p_stmt = isl_calloc_type(ctx, struct polysa_stmt); // n
  p_stmt->stmt = stmt; // n
  p_stmt->t_stmt = NULL;
  
	if (!stmt)
		goto error;
  if (!p_stmt)
    goto error;
	if (!stmt->stmt)
		goto error;

	map = isl_map_from_union_map(isl_ast_build_get_schedule(build));
//  // debug
//  isl_printer *pr = isl_printer_to_file(isl_ast_build_get_ctx(build), stdout);
//  pr = isl_printer_print_map(pr, map);
//  printf("\n");
//  // debug
	map = isl_map_reverse(map);
//  // debug
//  pr = isl_printer_print_map(pr, map);
//  printf("\n");
//  // debug

	iterator_map = isl_pw_multi_aff_from_map(map);
//  // debug
//  pr = isl_printer_print_pw_multi_aff(pr, iterator_map);
//  printf("\n");
//  // debug
	stmt->ref2expr = pet_stmt_build_ast_exprs(stmt->stmt, build,
				    &pullback_index, iterator_map, NULL, NULL);
//  // debug
//  pr = isl_printer_print_id_to_ast_expr(pr, stmt->ref2expr);
//  printf("\n");
//  // debug
	isl_pw_multi_aff_free(iterator_map);

  id = isl_id_alloc(isl_ast_node_get_ctx(node), "user", p_stmt);
	id = isl_id_set_free_user(id, &polysa_stmt_free);
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
  struct polysa_stmt *p_stmt;
	isl_id *id;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	p_stmt = isl_id_get_user(id);
  stmt = p_stmt->stmt;
	isl_id_free(id);

  if (stmt == NULL && p_stmt->t_stmt != NULL)
    return isl_bool_true;
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

  /* Set up printer to T2S. */
  FILE *t2s_fp = fopen("t2s.cpp.tmp", "w");
  isl_printer *t2s_p = isl_printer_to_file(ctx, t2s_fp);

	build = isl_ast_build_alloc(ctx);
	iterators = ppcg_scop_generate_names(scop, depth, "c");
	build = isl_ast_build_set_iterators(build, iterators);
	build = isl_ast_build_set_at_each_domain(build, &at_each_domain, scop); // TODO

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
  
//  // debug
//  isl_printer *printer = isl_printer_to_file(ctx, stdout);
//  isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
//  isl_printer_print_ast_node(printer, tree);
//  printf("\n");
//
//  isl_ast_node_foreach_descendant_top_down(tree,
//      &debug_ast_node, NULL);
//  isl_printer_free(printer);
//  // debug

	isl_ast_build_free(build);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_user, &t2s_p);

	print_options = isl_ast_print_options_set_print_for(print_options,
							&print_for, NULL);

	p = cpu_print_macros(p, tree);
  isl_printer *p_copy = p;
	p = isl_ast_node_print(tree, p_copy, print_options);

  isl_printer_free(t2s_p);
  fclose(t2s_fp);

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
static __isl_give isl_schedule_node *tile_band(
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
static __isl_give isl_schedule *get_schedule(struct ppcg_scop *ps,
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
							&tile_band, ps);

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

static __isl_give isl_union_set *create_t2s_domain(isl_schedule_node *node, char *content)
{
  isl_space *space;
  isl_id *id;
  char name[40];

  space = isl_space_set_alloc(isl_schedule_node_get_ctx(node), 0, 0);
  snprintf(name, sizeof(name), "t2s_stmt");
  id = isl_id_alloc(isl_schedule_node_get_ctx(node), name, content);
  space = isl_space_set_tuple_id(space, isl_dim_set, id);
  return isl_union_set_from_set(isl_set_universe(space));
}

static __isl_give isl_schedule_node *aggregate_stmt_domain(__isl_take isl_schedule_node *node, void *user)
{
  isl_union_set *domain;
  isl_union_map *schedule;
  isl_set *stmt_domain;
  isl_set **anchor_domain = (isl_set **)(user);

  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  domain = isl_schedule_node_get_domain(node);
  schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
  schedule = isl_union_map_intersect_domain(schedule, domain);
  stmt_domain = isl_set_from_union_set(isl_union_map_range(schedule));
  if (*anchor_domain == NULL)
    *anchor_domain = isl_set_copy(stmt_domain);
  else
    *anchor_domain = isl_set_union(*anchor_domain, isl_set_copy(stmt_domain));

  isl_set_free(stmt_domain);

  return node;
}

static __isl_give isl_schedule_node *gen_stmt_domain(__isl_take isl_schedule_node *node, void *user)
{
  struct t2s_data *data = (struct t2s_data *)(user);
  isl_union_set *stmt_domain;
  isl_union_pw_multi_aff *contraction;
  isl_union_map *stmt_schedule;
  isl_set *stmt_domain_i;
  isl_set *stmt_sim_domain_i;
  const char *stmt_name;
  isl_set *set;
  isl_space *space;

  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  stmt_domain = isl_schedule_node_get_domain(node);  
  contraction = isl_schedule_node_get_subtree_contraction(node);
  stmt_domain = isl_union_set_preimage_union_pw_multi_aff(stmt_domain, contraction);
  stmt_schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_print_union_map(p, stmt_schedule);
//  printf("\n");
//  // debug
  stmt_domain = isl_union_map_range(isl_union_map_intersect_domain(stmt_schedule, stmt_domain));
//  // debug
//  isl_printer_print_union_set(p, stmt_domain);
//  printf("\n");
//  // debug
  stmt_domain_i = isl_set_from_union_set(stmt_domain);
  stmt_sim_domain_i = isl_set_gist(isl_set_copy(stmt_domain_i), 
      isl_set_copy(data->anchor_domain));

  /* Set the name of space. */
  set = isl_set_from_union_set(isl_schedule_node_get_domain(node));
  space = isl_set_get_space(set);
  stmt_name = isl_space_get_tuple_name(space, isl_dim_set);
  stmt_domain_i = isl_set_set_tuple_name(stmt_domain_i, stmt_name);  
  stmt_sim_domain_i = isl_set_set_tuple_name(stmt_sim_domain_i, stmt_name);
  
//  // debug
//  isl_printer_print_set(p, stmt_domain_i);
//  printf("\n");
//  // debug
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

  // stmt_domain_simple = isl_set_gist(isl_set_from_union_set(stmt_domain), isl_set_copy(data->anchor_domain));

  return node;
}

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

/* This function extracts the raw and rar deps that has the dest access associated 
 * with the current access.
 */
static int t2s_update_dep(__isl_keep pet_expr *expr, void *user)
{
  struct t2s_data *data = user;
  struct t2s_stmt_data *stmt_data = data->stmt_data;

  isl_id *id;
  id = isl_id_copy(expr->acc.ref_id);

//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_print_id(p, id);
//  printf("\n");
//  // debug
 
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
//      // debug
//      isl_printer *p_d = isl_printer_to_file(data->ctx, stdout);
//      p_d = isl_printer_print_basic_map(p_d, dep_i->isl_dep);
//      printf("\n");
//      p_d = isl_printer_print_id(p_d, id);
//      printf("\n");
//      p_d = isl_printer_print_id(p_d, dep_i->dest);
//      printf("\n");
//      isl_printer_free(p_d);
//      // debug
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

  index_space = isl_multi_pw_aff_get_space(index);
  array_id = isl_space_get_tuple_id(index_space, isl_dim_out);

  isl_multi_pw_aff_free(index);
  isl_space_free(index_space);

  /* If the access is associated with RAR, then generate access as
   * A(c0, c1, c2).
   * If the access is associated with RAW, then generate access as
   * A(c0, c1 - 1, c2).
   * Otherwise, generate A(c0, c1, c2).
   */
  isl_id *func = isl_id_copy(array_id);
  isl_ast_expr *func_expr = isl_ast_expr_from_id(func);
  isl_ast_expr_list *args = isl_ast_expr_list_alloc(ctx, 0);

  isl_id_free(array_id);

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

void *polysa_dep_free(__isl_take struct polysa_dep *dep)
{
  if (!dep)
    return NULL;

  isl_id_free(dep->src);
  isl_id_free(dep->dest);
  isl_vec_free(dep->disvec);
  isl_set_free(dep->src_sched_domain);
  isl_set_free(dep->dest_sched_domain);
  isl_basic_map_free(dep->isl_dep);

  free(dep);

  return NULL;
}

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

static char *concat(const char *s1, const char *s2) 
{
  char *result = malloc(strlen(s1) + strlen(s2) + 1);
  strcpy(result, s1);
  strcat(result, s2);
  return result;
}

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
//      // debug
//      isl_printer *p_debug = isl_printer_to_file(isl_set_get_ctx(set), stdout);
//      p_debug = isl_printer_print_constraint(p_debug, cst_i);
//      printf("\n");
//      // debug

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
//  // debug
//  printf("%s\n", t2s_cst);
//  // debug
  isl_printer_free(p);

  return t2s_cst;
}

/* This function takes in the C statement like
 * C[i][j] = 0
 * and prints out the T2S statement like
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
        p = isl_printer_print_str(p, ", ");
        p = isl_printer_print_str(p, LHS_func);
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

//  free(d->dep_stmt_pair);

  free(d);
  
  return NULL;
}

/* Transform each user statement in the original program to a T2S URE. */
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
  stmt->stmt = find_stmt(data->scop, id);  
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

  data->stmt_data = stmt_data;
  /* Extract the ref2expr for each access. */
  extract_t2s_stmt_access(stmt, data);

  data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + data->stmt_data->stmt_num));
  /* Print the stmt to data->t2s_stmt_text and update data->t2s_stmt_num. */
  for (int i = 0; i < data->stmt_data->stmt_num; i++) {
    isl_printer *p_str = isl_printer_to_str(data->ctx);
	  p_str = isl_printer_set_output_format(p_str, ISL_FORMAT_C);
    struct ppcg_stmt *stmt_i = data->stmt_data->stmts[i];
    p_str = pet_stmt_print_body(stmt_i->stmt, p_str, stmt_i->ref2expr);
    data->t2s_stmt_text[data->t2s_stmt_num + i] = isl_printer_get_str(p_str);
    data->t2s_stmt_text[data->t2s_stmt_num + i] = c_to_t2s_stmt(data->t2s_stmt_text[data->t2s_stmt_num + i], 
        isl_set_copy(data->stmt_data->stmt_domain[i]), data->iter_num);
    isl_printer_free(p_str);
  }
  data->t2s_stmt_num += data->stmt_data->stmt_num;

  data->stmt_data = t2s_stmt_data_free(stmt_data);

  return node;
}

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

  for (n = 0; n < data->ndeps; n++) {
    dep = data->deps[n];
    if (dep->dest == id && dep->type == POLYSA_DEP_RAR)
      break;
  }

  if (n != data->ndeps) {
    isl_set *stmt_domain = isl_set_copy(stmt_data->stmt_anchor_domain);
    isl_set *dep_dest_domain = isl_set_copy(dep->dest_sched_domain);
    /* Project out the extra scalar dimensions. */
    dep_dest_domain = isl_set_project_out(dep_dest_domain, 
        isl_dim_set, data->iter_num, isl_set_dim(dep_dest_domain, isl_dim_set) - data->iter_num);
    dep_dest_domain = isl_set_set_tuple_name(dep_dest_domain, isl_set_get_tuple_name(stmt_domain));

    /* Generate the init domain */
    isl_set *init_domain = isl_set_subtract(stmt_domain, isl_set_copy(dep_dest_domain));
    isl_set *anchor_domain = isl_set_copy(data->anchor_domain);
    anchor_domain = isl_set_set_tuple_name(anchor_domain, isl_set_get_tuple_name(init_domain));
    init_domain = isl_set_gist(init_domain, isl_set_copy(anchor_domain));

    /* Set up the iterator names. */
    init_domain = t2s_set_set_iters(init_domain);
    char *init_domain_str = isl_set_to_t2s_format(init_domain);
    isl_set_free(init_domain);

    isl_set *reuse_domain = isl_set_gist(dep_dest_domain, anchor_domain);
    reuse_domain = t2s_set_set_iters(reuse_domain);
    char *reuse_domain_str = isl_set_to_t2s_format(reuse_domain);
    isl_set_free(reuse_domain);

    /* Generate the func name .*/
    isl_id *array_id;
    isl_space *index_space;
    index_space = isl_multi_pw_aff_get_space(index);
    array_id = isl_space_get_tuple_id(index_space, isl_dim_out);
    isl_space_free(index_space);
    
    isl_id *func = isl_id_copy(array_id);
    isl_id_free(array_id);
    isl_printer *p_str = isl_printer_to_str(ctx);
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
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, func_str);
    p_str = isl_printer_print_str(p_str, " = 0;\n");
    char *URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
    data->t2s_stmt_num++;

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
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, func_str);
    p_str = isl_printer_print_str(p_str, "));\n");
    URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
    data->t2s_stmt_num++;

    isl_id_free(func);
    free(func_str);
    free(init_domain_str);
    free(acc_str);
    free(reuse_domain_str);
    free(reuse_func_str);
  }

  isl_id_free(id);
  isl_multi_pw_aff_free(index);

  return 0;
}

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
  stmt->stmt = find_stmt(data->scop, id);
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
  isl_set *writeout_domain = NULL;

  for (n = 0; n < data->ndeps; n++) {
    dep = data->deps[n];
    if (dep->src == id && dep->type == POLYSA_DEP_WAW) {
      break;
    }
  }

  if (n != data->ndeps) {
    isl_set *stmt_domain = isl_set_copy(stmt_data->stmt_anchor_domain);
    isl_set *dep_src_domain = isl_set_copy(dep->src_sched_domain);
    /* Project out the extra scalar dimensions. */
    dep_src_domain = isl_set_project_out(dep_src_domain, 
        isl_dim_set, data->iter_num, isl_set_dim(dep_src_domain, isl_dim_set) - data->iter_num);
    dep_src_domain = isl_set_set_tuple_name(dep_src_domain, isl_set_get_tuple_name(stmt_domain));

    /* Generate the writeout domain */
    isl_set *writeout_domain = isl_set_subtract(stmt_domain, dep_src_domain);
    isl_set *anchor_domain = isl_set_copy(data->anchor_domain);
    anchor_domain = isl_set_set_tuple_name(anchor_domain, isl_set_get_tuple_name(writeout_domain));
    writeout_domain = isl_set_gist(writeout_domain, anchor_domain);

    if (isl_set_is_empty(writeout_domain)) {
      isl_set_free(writeout_domain);
      isl_id_free(id);
      isl_multi_pw_aff_free(index);

      return 0;
    }

    /* Set up the iterator names. */
    writeout_domain = t2s_set_set_iters(writeout_domain);
    char *writeout_domain_str = isl_set_to_t2s_format(writeout_domain);
    isl_set_free(writeout_domain);

    /* Generate the func name .*/
    isl_id *array_id;
    isl_space *index_space;
    index_space = isl_multi_pw_aff_get_space(index);
    array_id = isl_space_get_tuple_id(index_space, isl_dim_out);
    isl_space_free(index_space);
    
    isl_id *func = isl_id_copy(array_id);
    isl_id_free(array_id);
    isl_printer *p_str = isl_printer_to_str(ctx);
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
    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, drain_func_str);    
    p_str = isl_printer_print_str(p_str, " = 0;\n");
    char *URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
    data->t2s_stmt_num++;

    p_str = isl_printer_to_str(ctx);
    p_str = isl_printer_print_str(p_str, drain_func_str);
    p_str = isl_printer_print_str(p_str, " = ");
    p_str = isl_printer_print_str(p_str, "select(");
    p_str = isl_printer_print_str(p_str, writeout_domain_str);
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, func_str);
    p_str = isl_printer_print_str(p_str, ", ");
    p_str = isl_printer_print_str(p_str, drain_func_str);
    p_str = isl_printer_print_str(p_str, ");\n");
    URE_text = isl_printer_get_str(p_str);
    isl_printer_free(p_str);

    data->t2s_stmt_text = (char **)realloc(data->t2s_stmt_text, sizeof(char *) * (data->t2s_stmt_num + 1));
    data->t2s_stmt_text[data->t2s_stmt_num] = URE_text;
    data->t2s_stmt_num++;

    isl_id_free(func);
    free(func_str);
    free(drain_func_str);
    free(writeout_domain_str);
//    free(acc_str);
  }

  isl_id_free(id);
  isl_multi_pw_aff_free(index);

  return 0;
 
}

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
  stmt->stmt = find_stmt(data->scop, id);
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

static __isl_give isl_schedule *gen_stmt_text_wrap(__isl_take isl_schedule *schedule, struct t2s_data *data)
{
  /* Generate the reuse (RAR) statement. */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_op_stmt_text, data);

  /* Traverse each statmenet, build the ppcg_stmt struct and update
   * the ref2expr using T2S functions.
   * Print the stmt to data->t2s_stmt_text.
   */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_stmt_text, data);

  /* Generate the drain statement. */
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
    &gen_drain_stmt_text, data);

  /* Print out the T2S stmt texts. */
  for (int i = 0; i < data->t2s_stmt_num; i++) {
//    data->p = isl_printer_start_line(data->p);
    data->p = isl_printer_print_str(data->p, data->t2s_stmt_text[i]);
//    data->p = isl_printer_end_line(data->p);
  }
  data->p = isl_printer_start_line(data->p);
  data->p = isl_printer_end_line(data->p);

  return schedule;
}

static __isl_give isl_schedule_node *create_dep(__isl_take isl_schedule_node *node, void *user)
{
  struct t2s_data *data = user;
    
  if (!node)
    return NULL;

  if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
    return node;

  isl_union_map *dep_flow = data->scop->tagged_dep_flow;
  isl_union_map *dep_rar = data->scop->tagged_dep_rar;
  isl_union_map *dep = isl_union_map_union(dep_flow, dep_rar);
  
  isl_basic_map_list *deps = isl_union_map_get_basic_map_list(dep);
  int ndeps = isl_union_map_n_basic_map(dep);

  /* Add deps that has the dest as the current stmt. */

  return node;
}

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

  band = get_outermost_permutable_node(schedule);
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
    disvec = get_dep_dis_at_node(bmap_dep_i, band);
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
    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(src_domain));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(dest_domain));

    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));    
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));

    /* Remove the scalar dimensions. */
    p_dep_i->src_sched_domain = isl_set_project_out(p_dep_i->src_sched_domain, isl_dim_set, 
        data->iter_num, isl_set_dim(p_dep_i->src_sched_domain, isl_dim_set) - data->iter_num);
    p_dep_i->dest_sched_domain = isl_set_project_out(p_dep_i->dest_sched_domain, isl_dim_set,
        data->iter_num, isl_set_dim(p_dep_i->dest_sched_domain, isl_dim_set) - data->iter_num);

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
    disvec = get_dep_dis_at_node(bmap_dep_i, band);
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
    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(src_domain));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(dest_domain));
    
    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));
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
    disvec = get_dep_dis_at_node(bmap_dep_i, band);
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
    isl_union_map *sched_src = isl_union_map_intersect_domain(isl_union_map_copy(sched), isl_union_set_from_set(src_domain));
    isl_union_map *sched_dest = isl_union_map_intersect_domain(sched, isl_union_set_from_set(dest_domain));
    
    p_dep_i->src_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_src));
    p_dep_i->dest_sched_domain = isl_set_from_union_set(isl_union_map_range(sched_dest));
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

  isl_printer_free(d->p);
  for (int i = 0; i < d->ndeps; i++) {
    polysa_dep_free(d->deps[i]);    
  }
  free(d->deps);
  t2s_stmt_data_free(d->stmt_data); 

  free(d);

  return NULL;
}

static __isl_give isl_schedule *test_func0(__isl_take isl_schedule *schedule, __isl_keep struct ppcg_scop *scop)
{
  isl_multi_val *sizes = NULL;
  isl_space *space;
  isl_ctx *ctx = isl_schedule_get_ctx(schedule);

  isl_schedule_node *band = get_outermost_permutable_node(schedule);
  space = isl_schedule_node_band_get_space(band);
  // debug
  isl_printer *p = isl_printer_to_file(ctx, stdout);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_space(p, space);
  printf("\n");
  // debug

  sizes = isl_multi_val_zero(space);
  for (int i = 0; i < isl_space_dim(space, isl_dim_set); i++) {
    isl_val *v;
    v = isl_val_int_from_si(ctx, 8);
    sizes = isl_multi_val_set_val(sizes, i, v);
  }

  band = isl_schedule_node_band_tile(band, sizes);
  // debug
  p = isl_printer_print_schedule_node(p, band);
  printf("\n");
  // debug

  /* Trnasform the deps. */
  isl_union_map *deps = scop->dep_flow;
  isl_basic_map_list *deps_list = isl_union_map_get_basic_map_list(deps);
  isl_union_map *sched = isl_schedule_node_get_subtree_schedule_union_map(band);
  // debug  
  p = isl_printer_print_union_map(p, sched);
  printf("\n");
  // debug
  
  isl_basic_map_list *sched_list = isl_union_map_get_basic_map_list(sched);  

  for (int i = 0; i < isl_union_map_n_basic_map(deps); i++) {
    isl_basic_map *dep = isl_basic_map_list_get_basic_map(deps_list, i);
    // debug
    p = isl_printer_print_basic_map(p, dep);
    printf("\n");  
    // debug
    isl_basic_map *sched1 = isl_basic_map_list_get_basic_map(sched_list, 0); // S1
    isl_basic_map *sched0 = isl_basic_map_list_get_basic_map(sched_list, 1); // S0
    // debug
    p = isl_printer_print_basic_map(p, sched0);
    printf("\n");
    // debug
    
    dep = isl_basic_map_apply_domain(dep, sched0);
    dep = isl_basic_map_apply_range(dep, sched1);
    // debug
    p = isl_printer_print_basic_map(p, dep);
    printf("\n");
    // debug
  }

  isl_schedule_free(schedule);
  schedule = isl_schedule_node_get_schedule(band);

  isl_schedule_node_free(band);
  return schedule;
}

static __isl_give isl_schedule *test_func1(__isl_take isl_schedule *schedule, __isl_keep struct ppcg_scop *scop)
{
  isl_multi_val *sizes = NULL;
  isl_space *space;
  isl_ctx *ctx = isl_schedule_get_ctx(schedule);

  isl_schedule_node *band = get_outermost_permutable_node(schedule);
  space = isl_schedule_node_band_get_space(band);
  // debug
  isl_printer *p = isl_printer_to_file(ctx, stdout);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_space(p, space);
  printf("\n");
  // debug

//  sizes = isl_multi_val_zero(space);
//  for (int i = 0; i < isl_space_dim(space, isl_dim_set); i++) {
//    isl_val *v;
//    v = isl_val_int_from_si(ctx, 8);
//    sizes = isl_multi_val_set_val(sizes, i, v);
//  }
//
//  band = isl_schedule_node_band_tile(band, sizes);
  // debug
  p = isl_printer_print_schedule_node(p, band);
  printf("\n");
  // debug

  /* Trnasform the deps. */
  isl_union_map *deps = scop->dep_flow;
  isl_basic_map_list *deps_list = isl_union_map_get_basic_map_list(deps);

  for (int i = 0; i < isl_union_map_n_basic_map(deps); i++) {
    isl_basic_map *dep = isl_basic_map_list_get_basic_map(deps_list, i);
    // debug
    p = isl_printer_print_basic_map(p, dep);
    printf("\n");  
    // debug
  }

  isl_schedule_free(schedule);
  schedule = isl_schedule_node_get_schedule(band);

  isl_schedule_node_free(band);
  return schedule;
}


/* This function adds T2S statement as extension nodes after each original
 * user stmt. 
 */
static __isl_give isl_schedule *print_t2s_with_schedule(__isl_take isl_schedule *schedule, 
    __isl_keep struct ppcg_scop *scop)
{
  struct t2s_data *data;
  isl_union_set *stmt_domains;
  isl_union_map *stmt_schedules;
  isl_schedule_node *node;
  isl_union_set *stmt_trans_domains;
  isl_space *stmt_space;
  isl_ctx *ctx;

  ctx = isl_schedule_get_ctx(schedule);
  data = isl_calloc_type(ctx, struct t2s_data);

  data->ctx = ctx;
  data->scop = scop;
  FILE *t2s_fp = fopen("t2s.cpp", "w");
  data->p = isl_printer_to_file(ctx, t2s_fp); 

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
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
      &aggregate_stmt_domain, &data->anchor_domain);

  /* Calculate the simplified domain (in scheduling dims) for each statement. */
  data->stmt_domain = NULL;
  data->stmt_sim_domain = NULL;
  schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
      &gen_stmt_domain, data);

//  // debug
//  isl_printer *p = isl_printer_to_file(data->ctx, stdout);
//  p = isl_printer_print_union_set(p, data->stmt_domain);
//  printf("\n");
//  p = isl_printer_print_union_set(p, data->stmt_sim_domain);
//  printf("\n"); 
//  isl_printer_free(p);
//  // debug

  /* Generate the T2S statements .*/ 
  data->t2s_stmt_num = 0;
  data->t2s_stmt_text = NULL;
  schedule = gen_stmt_text_wrap(schedule, data);

  fclose(t2s_fp);
  t2s_data_free(data);

  return schedule;
}

/* Generate CPU code for "scop" and print it to "p".
 *
 * First obtain a schedule for "scop" and then print code for "scop"
 * using that schedule.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct ppcg_scop *scop, struct ppcg_options *options)
{
	isl_schedule *schedule;

	schedule = get_schedule(scop, options);

  /*  Check if the program is legal to be mapped to systolic array. */
  isl_bool is_legal = sa_legality_check(schedule, scop);
  if (is_legal != isl_bool_true) {
    printf("[PolySA] Illegal to be transformed to systolic array.\n");
  }

//  if (is_legal) {
//    /* Generate systolic arrays using space-time mapping. */
//    isl_size num_sa = 0;
//    struct polysa_prog **sa_candidates = sa_space_time_transform(schedule, scop, &num_sa);
//    if (num_sa > 0) {
//      printf("[PolySA] %d systolic arrays generated.\n", num_sa);
//    }
//
//    /* Pick up one systolic array to proceed based on heuristics. */
//    struct polysa_prog *sa_opt = sa_candidates_smart_pick(sa_candidates, num_sa);
//
//    /* Extract common VSA features. */
//    struct polysa_vsa *vsa = polysa_vsa_alloc();
//    vsa_band_width_extract(sa_opt, vsa);
//
//    /* Extract T2S features. */
//    if (options->target == POLYSA_TARGET_T2S) {
//      vsa_t2s_iter_extract(sa_opt, vsa);
//      vsa_t2s_var_extract(sa_opt, vsa);
//    }
//
//    polysa_vsa_free(vsa);
//
//    /* Apply PE optimization. */
//    sa_pe_optimize(sa_opt);
//
//    schedule = isl_schedule_copy(sa_opt->schedule);
//    polysa_prog_free(sa_opt);
//  }

  schedule = print_t2s_with_schedule(schedule, scop);

//  schedule = test_func0(schedule, scop);
//  schedule = test_func1(schedule, scop);

	return print_cpu_with_schedule(p, scop, schedule, options);
}

/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *print_polysa_cpu_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct ppcg_options *options = user;

	return generate(p, scop, options);
}

///* Derive the output file name from the input file name.
// * 'input' is the entire path of the input file. The output
// * is the file name plus the additional extension.
// *
// * We will basically replace everything after the last point
// * with '.ppcg.c'. This means file.c becomes file.ppcg.c
// */
//static FILE *get_output_file(const char *input, const char *output)
//{
//	char name[PATH_MAX];
//	const char *ext;
//	const char ppcg_marker[] = ".ppcg";
//	int len;
//	FILE *file;
//
//	len = ppcg_extract_base_name(name, input);
//
//	strcpy(name + len, ppcg_marker);
//	ext = strrchr(input, '.');
//	strcpy(name + len + sizeof(ppcg_marker) - 1, ext ? ext : ".c");
//
//	if (!output)
//		output = name;
//
//	file = fopen(output, "w");
//	if (!file) {
//		fprintf(stderr, "Unable to open '%s' for writing\n", output);
//		return NULL;
//	}
//
//	return file;
//}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CPU code and write the results to a file
 * called "output".
 */
int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *options, 
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
          &print_polysa_cpu_wrap, options);

  fclose(output_file);

  return r;
}

