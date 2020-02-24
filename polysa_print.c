#include "polysa_print.h"

/* Was the definition of "type" printed before?
 * That is, does its name appear in the list of printed types "types"?
 */
static int already_printed(struct polysa_types *types,
	struct pet_type *type)
{
	int i;

	for (i = 0; i < types->n; ++i)
		if (!strcmp(types->name[i], type->name))
			return 1;

	return 0;
}

/* Print the definitions of all types prog->scop that have not been
 * printed before (according to "types") on "p".
 * Extend the list of printed types "types" with the newly printed types.
 */
__isl_give isl_printer *polysa_print_types(__isl_take isl_printer *p, 
  struct polysa_types *types, struct polysa_prog *prog)
{
  int i, n;
  isl_ctx *ctx;
  char **name;

  n = prog->scop->pet->n_type;

  if (n == 0)
    return p;

  ctx = isl_printer_get_ctx(p);
  name = isl_realloc_array(ctx, types->name, char *, types->n + n);
	if (!name)
		return isl_printer_free(p);
	types->name = name;

	for (i = 0; i < n; ++i) {
		struct pet_type *type = prog->scop->pet->types[i];

		if (already_printed(types, type))
			continue;

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, type->definition);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);

		types->name[types->n++] = strdup(type->name);
	}

	return p;  
}

/* This function is called for each node in a PolySA AST.
 * In case of a user node, print the macro definitions required
 * for printing the AST expressions in the annotation, if any.
 * For other nodes, return true such that descendants are also
 * visited.
 *
 * In particular, for a kernel launch, print the macro definitions
 * needed for the grid size.
 * For a copy statement, print the macro definitions needed
 * for the two index expressions.
 * For an original user statement, print the macro definitions
 * needed for the substitutions.
 */
static isl_bool at_node(__isl_keep isl_ast_node *node, void *user)
{
	const char *name;
	isl_id *id;
	int is_kernel;
	struct polysa_kernel *kernel;
	struct polysa_kernel_stmt *stmt;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	if (!id)
		return isl_bool_false;

	name = isl_id_get_name(id);
	if (!name)
		return isl_bool_error;
	is_kernel = !strcmp(name, "kernel");
	kernel = is_kernel ? isl_id_get_user(id) : NULL;
	stmt = is_kernel ? NULL : isl_id_get_user(id);
	isl_id_free(id);

	if ((is_kernel && !kernel) || (!is_kernel && !stmt))
		return isl_bool_error;

	if (is_kernel) {
		*p = ppcg_ast_expr_print_macros(kernel->grid_size_expr, *p);
	} else if (stmt->type == POLYSA_KERNEL_STMT_COPY) {
		*p = ppcg_ast_expr_print_macros(stmt->u.c.index, *p);
		*p = ppcg_ast_expr_print_macros(stmt->u.c.local_index, *p);
	} else if (stmt->type == POLYSA_KERNEL_STMT_DOMAIN) {
		*p = ppcg_print_body_macros(*p, stmt->u.d.ref2expr);
	}
	if (!*p)
		return isl_bool_error;

	return isl_bool_false;
}

/* Print the required macros for the PolySA AST "node" to "p",
 * including those needed for the user statements inside the AST.
 */
__isl_give isl_printer *polysa_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_descendant_top_down(node, &at_node, &p) < 0)
		return isl_printer_free(p); 
	p = ppcg_print_macros(p, node); 
	return p;
}

/* Print the declaration of a non-linearized array argument.
 */
static __isl_give isl_printer *print_non_linearized_declaration_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");

	p = isl_printer_print_ast_expr(p, array->bound_expr);

	return p;
}

/* Print the declaration of an array argument.
 * "memory_space" allows to specify a memory space prefix.
 */
__isl_give isl_printer *polysa_array_info_print_declaration_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array,
	const char *memory_space)
{
	if (polysa_array_is_read_only_scalar(array)) {
		p = isl_printer_print_str(p, array->type);
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	if (memory_space) {
		p = isl_printer_print_str(p, memory_space);
		p = isl_printer_print_str(p, " ");
	}

	if (array->n_index != 0 && !array->linearize)
		return print_non_linearized_declaration_argument(p, array);

	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "*");
	p = isl_printer_print_str(p, array->name);

	return p;
}

__isl_give isl_printer *polysa_kernel_print_domain(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt)
{
	return pet_stmt_print_body(stmt->u.d.stmt->stmt, p, stmt->u.d.ref2expr);
}

/* Print declarations to "p" for arrays that are local to "prog"
 * but that are used on the host and therefore require a declaration.
 */
__isl_give isl_printer *polysa_print_local_declarations(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	int i;

	if (!prog)
		return isl_printer_free(p);

	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];
		isl_ast_expr *size;

		if (!array->declare_local)
			continue;
		size = array->declared_size;
		p = ppcg_print_declaration_with_size(p, array->type, size);
	}

	return p;
}

/* Print an expression for the size of "array" in bytes.
 */
__isl_give isl_printer *polysa_array_info_print_size(__isl_take isl_printer *prn,
	struct polysa_array_info *array)
{
	int i;

	for (i = 0; i < array->n_index; ++i) {
		isl_ast_expr *bound;

		prn = isl_printer_print_str(prn, "(");
		bound = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
		prn = isl_printer_print_ast_expr(prn, bound);
		isl_ast_expr_free(bound);
		prn = isl_printer_print_str(prn, ") * ");
	}
	prn = isl_printer_print_str(prn, "sizeof(");
	prn = isl_printer_print_str(prn, array->type);
	prn = isl_printer_print_str(prn, ")");

	return prn;
}

/* Print an expression for the size of "array" in data items.
 */
__isl_give isl_printer *polysa_array_info_print_data_size(__isl_take isl_printer *prn,
	struct polysa_array_info *array)
{
	int i;
  int first = 1;

	for (i = 0; i < array->n_index; ++i) {
    if (!first)
      prn = isl_printer_print_str(prn, " * ");

		isl_ast_expr *bound;

		prn = isl_printer_print_str(prn, "(");
		bound = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
		prn = isl_printer_print_ast_expr(prn, bound);
		isl_ast_expr_free(bound);
		prn = isl_printer_print_str(prn, ")");
    first = 0;
	}

	return prn;
}
