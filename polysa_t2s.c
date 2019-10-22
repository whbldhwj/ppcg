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
#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "print.h"
#include "schedule.h"
#include "util.h"
#include "polysa_t2s.h"

/* Open the cpu .c file and the t2s .cpp file for writing.
 * Add the necessary includes.
 */
void t2s_open_files(struct t2s_info *info, const char *input)
{
  char name[PATH_MAX];
  int len;

  len = ppcg_extract_base_name(name, input);

  strcpy(name + len, "_cpu.c");
  info->host_c = fopen(name, "w");

  strcpy(name + len, "_t2s.cpp");
  info->kernel_c = fopen(name, "w");

  fprintf(info->kernel_c, "#include \"Halide.h\"\n");
  fprintf(info->kernel_c, "#include <iostream>\n\n");
  fprintf(info->kernel_c, "using namespace Halide;\n");
  fprintf(info->kernel_c, "using namespace std;\n");

//    fprintf(info->host_c, "#include <assert.h>\n");
//    fprintf(info->host_c, "#include <stdio.h>\n");
//    fprintf(info->host_c, "#include \"%s\"\n", name);
//    fprintf(info->kernel_c, "#include \"%s\"\n", name);
//    fprintf(info->kernel_h, "#include \"cuda.h\"\n\n");
}

/* Close all output files.
 */
void t2s_close_files(struct t2s_info *info)
{
  fclose(info->host_c);
  fclose(info->kernel_c);
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
	struct gpu_prog *prog, struct ppcg_kernel *kernel, int types)
{
	int i, n;
	int first = 1;
	unsigned nparam;
	isl_space *space;
	const char *type;

	for (i = 0; i < prog->n_array; ++i) {
		int required;

		required = ppcg_kernel_requires_array_argument(kernel, i);
		if (required < 0)
			return isl_printer_free(p);
		if (!required)
			continue;

		if (!first)
			p = isl_printer_print_str(p, ", ");

		if (types)
			p = gpu_array_info_print_declaration_argument(p,
				&prog->array[i], NULL);
		else
			p = gpu_array_info_print_call_argument(p,
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

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *print_kernel_header(__isl_take isl_printer *p,
	struct gpu_prog *prog, struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "void kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "(");
	p = print_kernel_arguments(p, prog, kernel, 1);
	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the header of the given kernel to both gen->t2s.kernel_h
 * and gen->t2s.kernel_c.
 */
static void print_kernel_headers(struct gpu_prog *prog,
	struct ppcg_kernel *kernel, struct t2s_info *t2s)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, t2s->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	p = isl_printer_to_file(prog->ctx, t2s->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

static __isl_give isl_printer *print_kernel_var(__isl_take isl_printer *p,
	struct ppcg_kernel_var *var)
{
	int j;

	p = isl_printer_start_line(p);
	if (var->type == ppcg_access_shared)
		p = isl_printer_print_str(p, "__shared__ ");
	p = isl_printer_print_str(p, var->array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
	for (j = 0; j < var->array->n_index; ++j) {
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
	struct ppcg_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = print_kernel_var(p, &kernel->var[i]);

	return p;
}

/* Print a sync statement.
 */
static __isl_give isl_printer *print_sync(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__syncthreads();");
	p = isl_printer_end_line(p);

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
	struct ppcg_kernel_stmt *stmt;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	isl_ast_print_options_free(print_options);

	switch (stmt->type) {
	case ppcg_kernel_copy:
		return ppcg_kernel_print_copy(p, stmt);
	case ppcg_kernel_sync:
		return print_sync(p, stmt);
	case ppcg_kernel_domain:
		return ppcg_kernel_print_domain(p, stmt);
	}

	return p;
}

static void print_kernel(struct gpu_prog *prog, struct ppcg_kernel *kernel, 
    struct t2s_info *t2s)
{
  isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
  isl_ast_print_options *print_options;
  isl_printer *p;

  print_kernel_headers(prog, kernel, t2s);
  fprintf(t2s->kernel_c, "{\n");

  p = isl_printer_to_file(ctx, t2s->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_indent(p, 4);

  p = print_kernel_vars(p, kernel); // TODO
  p = isl_printer_end_line(p);
  p = ppcg_set_macro_names(p);
  p = gpu_print_macros(p, kernel->tree);

  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                &print_kernel_stmt, NULL);
  p = isl_ast_node_print(kernel->tree, p, print_options);
  isl_printer_free(p);

  fprintf(t2s->kernel_c, "}\n");
}

/* Printer the user statement of the host code to "p".
 *
 * The host code may contain original user statements, kernel launches,
 * statements that copy data to/from the device and statements
 * that initialize or clear the device.
 * The original user satements and the kernel launches have an 
 * associated annotation, while the other statements do not.
 * The latter are handled by print_device_code.
 * The annotation on the user statements is called "user".
 */
static __isl_give isl_printer *print_host_user(__isl_take isl_printer *p,
    __isl_take isl_ast_print_options *print_options,
    __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  int is_user;
  struct ppcg_kernel *kernel;
  struct ppcg_kernel_stmt *stmt;
  struct print_host_user_data *data;

  isl_ast_print_options_free(print_options);

  data = (struct print_host_user_data *) user;

  id = isl_ast_node_get_annotation(node);
  
  is_user = !strcmp(isl_id_get_name(id), "user");
  kernel = is_user ? NULL : isl_id_get_user(id);
  stmt = is_user ? isl_id_get_user(id) : NULL;
  isl_id_free(id);

  if (is_user)
    return ppcg_kernel_print_domain(p, stmt);

  p = ppcg_start_block(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "kernel");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "(");
  p = print_kernel_arguments(p, data->prog, kernel, 0);
  p = isl_printer_print_str(p, ")");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p);

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  print_kernel(data->prog, kernel, data->t2s);

  return p;
}

static __isl_give isl_printer *print_host_code(__isl_take isl_printer *p,
    struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
    struct t2s_info *t2s)
{
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_ast_node_get_ctx(tree);
  struct print_host_user_data data = {t2s, prog};

  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
      &print_host_user, &data);

  p = gpu_print_macros(p, tree);
  p = isl_ast_node_print(tree, p, print_options);
}

/* Given a systolic array embedded in gpu_prog "prog" and the 
 * corresponding transformed AST "tree", print the entire T2S code to "p".
 * "types" colelcts the types for which a definition has already been printed.
 */
static __isl_give isl_printer *print_t2s(__isl_take isl_printer *p,
		struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		struct gpu_types *types, void *user) {
  struct t2s_info *t2s = user;
  isl_printer *kernel;

  kernel = isl_printer_to_file(isl_printer_get_ctx(p), t2s->kernel_c);
  kernel = isl_printer_set_output_format(kernel, ISL_FORMAT_C);
  kernel = gpu_print_types(kernel, types, prog);
  isl_printer_free(kernel);

  if (!kernel)
    return isl_printer_free(p);

  p = print_host_code(p, prog, tree, t2s);

  return p;
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding T2S code.
 * The names of the output files are derived from "input".
 *
 * We let generate_gpu do all the hard work and then let it call
 * us back for printing the AST in print_cuda.
 *
 * To prepare for this printing, we first open the output files
 * and we close them after generate_gpu has finished.
 */
int generate_polysa_t2s(isl_ctx *ctx, struct ppcg_options *options,
	const char *input)
{
	struct t2s_info t2s;
	int r;

	t2s_open_files(&t2s, input);

  r = generate_sa(ctx, input, t2s.host_c, options, &print_t2s, &t2s);

	t2s_close_files(&t2s);

	return r;
}
