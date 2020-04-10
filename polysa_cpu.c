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
#include "polysa_cpu.h"

struct cpu_info {
  FILE *host_c;          /* C host. */
  FILE *kernel_c;        /* Definition of hardware modules. */
  FILE *kernel_h;        /* Declaration of hardware modules. */
};

struct print_host_user_data {
	struct cpu_info *cpu;
	struct polysa_prog *prog;
};

/* Open the host .c file and the kernel .h and .c files for writing.
 * Add the necessary includes.
 */
static void cpu_open_files(struct cpu_info *info, const char *input)
{
  char name[PATH_MAX];
  int len;

  len = ppcg_extract_base_name(name, input);

  strcpy(name + len, "_host.c");
  info->host_c = fopen(name, "w");

  strcpy(name + len, "_kernel.c");
  info->kernel_c = fopen(name, "w");

  strcpy(name + len, "_kernel.h");
  info->kernel_h = fopen(name, "w");

  fprintf(info->host_c, "#include <assert.h>\n");
  fprintf(info->host_c, "#include <stdio,h>\n");
  fprintf(info->host_c, "#include \"%s\"\n", name);  
  fprintf(info->kernel_c, "#include \"%s\"\n", name);  
}

/* Print an access to the element in the private/shared memory copy
 * described by "stmt".  The index of the copy is recorded in
 * stmt->local_index as an access to the array.
 */
static __isl_give isl_printer *stmt_print_local_index(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt)
{
	return isl_printer_print_ast_expr(p, stmt->u.c.local_index);
}

static __isl_give isl_printer *io_stmt_print_local_index(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt)
{
	return isl_printer_print_ast_expr(p, stmt->u.i.local_index);
}

/* Print an access to the element in the global memory copy
 * described by "stmt".  The index of the copy is recorded in
 * stmt->index as an access to the array.
 */
static __isl_give isl_printer *stmt_print_global_index(
	__isl_take isl_printer *p, struct polysa_kernel_stmt *stmt)
{
	struct polysa_array_info *array = stmt->u.c.array;
	isl_ast_expr *index;

	if (polysa_array_is_scalar(array)) {
		if (!polysa_array_is_read_only_scalar(array))
			p = isl_printer_print_str(p, "*");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	index = isl_ast_expr_copy(stmt->u.c.index);

	p = isl_printer_print_ast_expr(p, index);
	isl_ast_expr_free(index);

	return p;
}

/* Print a copy statement.
 *
 * A read copy statement is printed as
 *
 *	local = global;
 *
 * while a write copy statement is printed as
 *
 *	global = local;
 */
static __isl_give isl_printer *polysa_kernel_print_copy(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	if (stmt->u.c.read) {
		p = stmt_print_local_index(p, stmt); 
		p = isl_printer_print_str(p, " = ");
		p = stmt_print_global_index(p, stmt);
	} else {
		p = stmt_print_global_index(p, stmt);
		p = isl_printer_print_str(p, " = ");
		p = stmt_print_local_index(p, stmt);
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

/* Print the call of an array argument.
 */
static __isl_give isl_printer *polysa_array_info_print_call_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
	if (polysa_array_is_read_only_scalar(array))
		return isl_printer_print_str(p, array->name);

	p = isl_printer_print_str(p, array->name);

	return p;
}

/* Print the arguments to a kernel declaration or call.  If "types" is set,
 * then print a declaration (including the types of the arguments).
 *
 * The arguments are printed in the following order
 * - the arrays accessed by the kernel
 * - the parameters
 * - the host loop iterators
 */
static __isl_give isl_printer *print_kernel_arguments(__isl_take isl_printer *p,
	struct polysa_prog *prog, struct polysa_kernel *kernel, int types)
{
	int i, n;
	int first = 1;
	unsigned nparam;
	isl_space *space;
	const char *type;

	for (i = 0; i < prog->n_array; ++i) {
		int required;

		required = polysa_kernel_requires_array_argument(kernel, i);
		if (required < 0)
			return isl_printer_free(p);
		if (!required)
			continue;

		if (!first)
			p = isl_printer_print_str(p, ", ");

		if (types)
			p = polysa_array_info_print_declaration_argument(p,
				&prog->array[i], 1, NULL, -1);
		else
			p = polysa_array_info_print_call_argument(p,
				&prog->array[i]);

		first = 0;
	}

	space = isl_union_set_get_space(kernel->arrays);
	nparam = isl_space_dim(space, isl_dim_param);
	for (i = 0; i < nparam; ++i) {
		const char *name;

		name = isl_space_get_dim_name(space, isl_dim_param, i);

		if (!first)
			p = isl_printer_print_str(p, ", ");
		if (types)
			p = isl_printer_print_str(p, "int ");
		p = isl_printer_print_str(p, name);

		first = 0;
	}
	isl_space_free(space);

	n = isl_space_dim(kernel->space, isl_dim_set);
	type = isl_options_get_ast_iterator_type(prog->ctx);
	for (i = 0; i < n; ++i) {
		const char *name;

		if (!first)
			p = isl_printer_print_str(p, ", ");
		name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
		if (types) {
			p = isl_printer_print_str(p, type);
			p = isl_printer_print_str(p, " ");
		}
		p = isl_printer_print_str(p, name);

		first = 0;
	}

	return p;
}

/* This function is called for each user statement in the AST,
 * i.e., for each kernel body statement, copy statement or sync statement.
 */
static __isl_give isl_printer *print_kernel_stmt(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	struct polysa_kernel_stmt *stmt;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	isl_ast_print_options_free(print_options);

	switch (stmt->type) {
	case POLYSA_KERNEL_STMT_COPY:
		return polysa_kernel_print_copy(p, stmt); 
//	case POLYSA_KERNEL_STMT_SYNC: 
//		return print_sync(p, stmt);
	case POLYSA_KERNEL_STMT_DOMAIN:
		return polysa_kernel_print_domain(p, stmt); 
	}

	return p;
}

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *print_kernel_header(__isl_take isl_printer *p,
	struct polysa_prog *prog, struct polysa_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "void kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "(");
	p = print_kernel_arguments(p, prog, kernel, 1);
	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the header of the given kernel to both gen->hls.kernel_h
 * and gen->hls.kernel_c.
 */
static void print_kernel_headers(struct polysa_prog *prog,
	struct polysa_kernel *kernel, struct cpu_info *cpu)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, cpu->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	p = isl_printer_to_file(prog->ctx, cpu->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* CPU Kernel Definition */");
  p = isl_printer_end_line(p);
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

static __isl_give isl_printer *print_kernel_var(__isl_take isl_printer *p,
	struct polysa_kernel_var *var)
{
	int j;

	p = isl_printer_start_line(p);
//	if (var->type == POLYSA_ACCESS_LOCAL)
//		p = isl_printer_print_str(p, "__local__ ");
	p = isl_printer_print_str(p, var->array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
	for (j = 0; j < isl_vec_size(var->size); ++j) {
		isl_val *v;

		p = isl_printer_print_str(p, "[");
		v = isl_vec_get_element_val(var->size, j);
		p = isl_printer_print_val(p, v);
		isl_val_free(v);
		p = isl_printer_print_str(p, "]");
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *print_kernel_vars(__isl_take isl_printer *p,
	struct polysa_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = print_kernel_var(p, &kernel->var[i]);

	return p;
}

static void print_kernel(struct polysa_prog *prog, struct polysa_kernel *kernel,
	struct cpu_info *cpu)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	isl_ast_print_options *print_options;
	isl_printer *p;

	print_kernel_headers(prog, kernel, cpu); 
	fprintf(cpu->kernel_c, "{\n");

	p = isl_printer_to_file(ctx, cpu->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_indent(p, 4);

	p = print_kernel_vars(p, kernel);
	p = isl_printer_end_line(p);
	p = ppcg_set_macro_names(p);
	p = polysa_print_macros(p, kernel->tree);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
						    &print_kernel_stmt, NULL); 
	p = isl_ast_node_print(kernel->tree, p, print_options);
	isl_printer_free(p);

	fprintf(cpu->kernel_c, "}\n");
  fprintf(cpu->kernel_c, "/* CPU Kernel Definition */\n\n");
}

/* Print the user statement of the host code to "p".
 *
 * The host code may contain original user statements, kernel launches,
 * statements that copy data to/from the device and statements
 * the initialize or clear the device.
 * The original user statements and the kernel launches have
 * an associated annotation, while the other statements do not.
 * The latter are handled by print_device_node.
 * The annotation on the user statements is called "user".
 *
 * In case of a kernel launch, print a block of statements that
 * defines the grid and the block and then launches the kernel.
 */
static __isl_give isl_printer *print_host_user(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  int is_user;
  struct polysa_kernel *kernel;
  struct polysa_kernel_stmt *stmt;
  struct print_host_user_data *data;

  isl_ast_print_options_free(print_options);

  data = (struct print_host_user_data *) user;

  id = isl_ast_node_get_annotation(node);
  if (!id)
    return p;

  is_user = !strcmp(isl_id_get_name(id), "user");
  kernel = is_user ? NULL : isl_id_get_user(id);
  stmt = is_user ? isl_id_get_user(id) : NULL;
  isl_id_free(id);

  if (is_user)
    return polysa_kernel_print_domain(p, stmt); 

  p = ppcg_start_block(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "kernel");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "(");
  p = print_kernel_arguments(p, data->prog, kernel, 0); 
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  print_kernel(data->prog, kernel, data->cpu); 

  return p;
}

static __isl_give isl_printer *polysa_print_host_code(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct cpu_info *cpu)
{
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_ast_node_get_ctx(tree);
  struct print_host_user_data data = { cpu, prog };
  isl_printer *p_module;

  /* Print the default AST. */
  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                &print_host_user, &data); 

  /* Print the macros definitions in the program. */
  p = polysa_print_macros(p, tree); 
  p = isl_ast_node_print(tree, p, print_options);
    
  return p;
}

/* Given a polysa_prog "prog" and the corresponding tranformed AST
 * "tree", print the entire OpenCL/HLS code to "p".
 * "types" collecs the types for which a definition has already been
 * printed.
 */
static __isl_give isl_printer *polysa_print_cpu(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct polysa_hw_top_module *top_module,
  struct polysa_types *types, void *user)
{
  struct cpu_info *cpu = user;
  isl_printer *kernel;

  kernel = isl_printer_to_file(isl_printer_get_ctx(p), cpu->kernel_c);
  kernel = isl_printer_set_output_format(kernel, ISL_FORMAT_C);
  kernel = polysa_print_types(kernel, types, prog);
  isl_printer_free(kernel);

  if (!kernel)
    return isl_printer_free(p);

  /* Print C host and kernel function */
  p = polysa_print_host_code(p, prog, tree, modules, n_modules, cpu); 

  return p;
}

/* Close all output files.
 */
static void cpu_close_files(struct cpu_info *info)
{
  fclose(info->kernel_c);
  fclose(info->kernel_h);
  fclose(info->host_c);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding HLS code.
 * The names of the output files are derived from "input".
 * 
 * We let generate_sa do all the hard work and then let it call 
 * us back for printing the AST in print_hls.
 * 
 * To prepare for the printing, we first open the output files
 * and we close them after generate_fpga has finished.
 */
int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input) 
{
  struct cpu_info cpu;
  int r;

  cpu_open_files(&cpu, input);

  r = generate_sa(ctx, input, cpu.host_c, options, &polysa_print_cpu, &cpu);

  cpu_close_files(&cpu);
}

