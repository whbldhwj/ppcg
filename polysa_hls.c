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
#include "polysa_hls.h"
#include "polysa_print.h"

struct hls_info {
  FILE *host_c;
  FILE *kernel_c;
  FILE *kernel_h;
};

struct print_host_user_data {
	struct hls_info *hls;
	struct polysa_prog *prog;
};

struct print_hw_module_data {
  struct hls_info *hls;
  struct polysa_prog *prog;
};

static __isl_give isl_printer *print_cuda_macros(__isl_take isl_printer *p)
{
	const char *macros =
		"#define cudaCheckReturn(ret) \\\n"
		"  do { \\\n"
		"    cudaError_t cudaCheckReturn_e = (ret); \\\n"
		"    if (cudaCheckReturn_e != cudaSuccess) { \\\n"
		"      fprintf(stderr, \"CUDA error: %s\\n\", "
		"cudaGetErrorString(cudaCheckReturn_e)); \\\n"
		"      fflush(stderr); \\\n"
		"    } \\\n"
		"    assert(cudaCheckReturn_e == cudaSuccess); \\\n"
		"  } while(0)\n"
		"#define cudaCheckKernel() \\\n"
		"  do { \\\n"
		"    cudaCheckReturn(cudaGetLastError()); \\\n"
		"  } while(0)\n\n";

	p = isl_printer_print_str(p, macros);
	return p;
}

/* Open the host .c file and the kernel .h and .c files for writing.
 * Add the necessary includes.
 */
static void hls_open_files(struct hls_info *info, const char *input)
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

/* Close all output files.
 */
static void hls_close_files(struct hls_info *info)
{
  fclose(info->kernel_c);
  fclose(info->kernel_h);
  fclose(info->host_c);
}

/* Does "kernel" need to be passed an argument corresponding to array "i"?
 *
 * The argument is only needed if the kernel accesses this device memory.
 */
static int polysa_kernel_requires_array_argument(struct polysa_kernel *kernel, int i)
{
	return kernel->array[i].global;
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
				&prog->array[i], NULL);
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

/* Print the effective grid size as a list of the sizes in each
 * dimension, from innermost to outermost.
 */
static __isl_give isl_printer *print_grid_size(__isl_take isl_printer *p,
	struct polysa_kernel *kernel)
{
	int i;
	int dim;

	dim = isl_multi_pw_aff_dim(kernel->grid_size, isl_dim_set);
	if (dim == 0)
		return p;

	p = isl_printer_print_str(p, "(");
	for (i = dim - 1; i >= 0; --i) {
		isl_ast_expr *bound;

		bound = isl_ast_expr_get_op_arg(kernel->grid_size_expr, 1 + i);
		p = isl_printer_print_ast_expr(p, bound);
		isl_ast_expr_free(bound);

		if (i > 0)
			p = isl_printer_print_str(p, ", ");
	}

	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the grid definition.
 */
static __isl_give isl_printer *print_grid(__isl_take isl_printer *p,
	struct polysa_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dim3 k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimGrid");
	p = print_grid_size(p, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static void print_reverse_list(FILE *out, int len, int *list)
{
	int i;

	if (!out || len == 0)
		return;

	fprintf(out, "(");
	for (i = 0; i < len; ++i) {
		if (i)
			fprintf(out, ", ");
		fprintf(out, "%d", list[len - 1 - i]);
	}
	fprintf(out, ")");
}

/* Print a declaration for the device array corresponding to "array" on "p".
 */
static __isl_give isl_printer *declare_device_array(__isl_take isl_printer *p,
	struct polysa_array_info *array)
{
	int i;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");
	if (!array->linearize && array->n_index > 1)
		p = isl_printer_print_str(p, "(");
	p = isl_printer_print_str(p, "*dev_");
	p = isl_printer_print_str(p, array->name);
	if (!array->linearize && array->n_index > 1) {
		p = isl_printer_print_str(p, ")");
		for (i = 1; i < array->n_index; i++) {
			isl_ast_expr *bound;
			bound = isl_ast_expr_get_op_arg(array->bound_expr,
							1 + i);
			p = isl_printer_print_str(p, "[");
			p = isl_printer_print_ast_expr(p, bound);
			p = isl_printer_print_str(p, "]");
			isl_ast_expr_free(bound);
		}
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *declare_device_arrays(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!polysa_array_requires_device_allocation(&prog->array[i]))
			continue;

		p = declare_device_array(p, &prog->array[i]);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *allocate_device_arrays(
	__isl_take isl_printer *p, struct polysa_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];

		if (!polysa_array_requires_device_allocation(&prog->array[i]))
			continue;
		p = ppcg_ast_expr_print_macros(array->bound_expr, p);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,
			"cudaCheckReturn(cudaMalloc((void **) &dev_");
		p = isl_printer_print_str(p, prog->array[i].name);
		p = isl_printer_print_str(p, ", ");
		p = polysa_array_info_print_size(p, &prog->array[i]);
		p = isl_printer_print_str(p, "));");
		p = isl_printer_end_line(p);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

/* Print code for initializing the device for execution of the transformed
 * code.  This includes declaring locally defined variables as well as
 * declaring and allocating the required copies of arrays on the device.
 */
static __isl_give isl_printer *init_device(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	p = print_cuda_macros(p); // TODO: remove in the future

	p = polysa_print_local_declarations(p, prog);
	p = declare_device_arrays(p, prog);
	p = allocate_device_arrays(p, prog);

	return p;
}

static __isl_give isl_printer *free_device_arrays(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!polysa_array_requires_device_allocation(&prog->array[i]))
			continue;
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaCheckReturn(cudaFree(dev_");
		p = isl_printer_print_str(p, prog->array[i].name);
		p = isl_printer_print_str(p, "));");
		p = isl_printer_end_line(p);
	}

	return p;
}

/* Print code for clearing the device after execution of the transformed code.
 * In particular, free the memory that was allocated on the device.
 */
static __isl_give isl_printer *clear_device(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	p = free_device_arrays(p, prog);

	return p;
}

/* Print code to "p" for copying "array" from the host to the device
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_to_device(__isl_take isl_printer *p,
	struct polysa_array_info *array)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	if (polysa_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	p = polysa_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyHostToDevice));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print code to "p" for copying "array" back from the device to the host
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * polysa_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_from_device(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(");
	if (polysa_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = polysa_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyDeviceToHost));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print a statement for copying an array to or from the device,
 * or for initializing or clearing the device.
 * The statement identifier of a copying node is called
 * "to_device_<array name>" or "from_device_<array name>" and
 * its user pointer points to the polysa_array_info of the array
 * that needs to be copied.
 * The node for initializing the device is called "init_device".
 * The node for clearing the device is called "clear_device".
 *
 * Extract the array (if any) from the identifier and call
 * init_device, clear_device, copy_array_to_device or copy_array_from_device.
 */
static __isl_give isl_printer *print_device_node(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node, struct polysa_prog *prog)
{
	isl_ast_expr *expr, *arg;
	isl_id *id;
	const char *name;
	struct polysa_array_info *array;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	array = isl_id_get_user(id);
	isl_id_free(id);
	isl_ast_expr_free(arg);
	isl_ast_expr_free(expr);

	if (!name)
		return isl_printer_free(p);
	if (!strcmp(name, "init_device"))
		return init_device(p, prog); 
	if (!strcmp(name, "clear_device"))
		return clear_device(p, prog); 
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array_to_device(p, array); 
	else
		return copy_array_from_device(p, array); 
}

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *print_kernel_header(__isl_take isl_printer *p,
	struct polysa_prog *prog, struct polysa_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__global__ void kernel");
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
	struct polysa_kernel *kernel, struct hls_info *hls)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, hls->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	p = isl_printer_to_file(prog->ctx, hls->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

static void print_indent(FILE *dst, int indent)
{
	fprintf(dst, "%*s", indent, "");
}

/* Print a list of iterators of type "type" with names "ids" to "out".
 * Each iterator is assigned one of the cuda identifiers in cuda_dims.
 * In particular, the last iterator is assigned the x identifier
 * (the first in the list of cuda identifiers).
 */
static void print_iterators(FILE *out, const char *type,
	__isl_keep isl_id_list *ids, const char *cuda_dims[])
{
	int i, n;

	n = isl_id_list_n_id(ids);
	if (n <= 0)
		return;
	print_indent(out, 4);
	fprintf(out, "%s ", type);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		if (i)
			fprintf(out, ", ");
		id = isl_id_list_get_id(ids, i);
		fprintf(out, "%s = %s", isl_id_get_name(id),
			cuda_dims[n - 1 - i]);
		isl_id_free(id);
	}
	fprintf(out, ";\n");
}

static void print_kernel_iterators(FILE *out, struct polysa_kernel *kernel)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	const char *type;
	const char *block_dims[] = { "blockIdx.x", "blockIdx.y" };
	const char *thread_dims[] = { "threadIdx.x", "threadIdx.y",
					"threadIdx.z" };

	type = isl_options_get_ast_iterator_type(ctx);

	print_iterators(out, type, kernel->block_ids, block_dims);
	print_iterators(out, type, kernel->thread_ids, thread_dims);
}

static __isl_give isl_printer *print_kernel_var(__isl_take isl_printer *p,
	struct polysa_kernel_var *var)
{
	int j;

	p = isl_printer_start_line(p);
	if (var->type == POLYSA_ACCESS_LOCAL)
		p = isl_printer_print_str(p, "__local__ ");
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
	struct polysa_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = print_kernel_var(p, &kernel->var[i]);

	return p;
}

/* Print a sync statement.
 */
static __isl_give isl_printer *print_sync(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__syncthreads();");
	p = isl_printer_end_line(p);

	return p;
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
__isl_give isl_printer *polysa_kernel_print_copy(__isl_take isl_printer *p,
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

/* Print an I/O statement.
 *
 * An in I/O statement is printed as
 *
 *  local = fifo.read();
 *
 * while an out I/O statement is printed as
 *
 *  fifo.write(local);
 */
__isl_give isl_printer *polysa_kernel_print_io(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt)
{
  p = isl_printer_start_line(p);
  if (stmt->u.i.in) {
    p = stmt_print_local_index(p, stmt);
    p = isl_printer_print_str(p, " = ");
    printf("%s\n", stmt->u.i.fifo_name);
    p = isl_printer_print_str(p, stmt->u.i.fifo_name);
    p = isl_printer_print_str(p, ".read()");
  } else {
    p = isl_printer_print_str(p, stmt->u.i.fifo_name);
    p = isl_printer_print_str(p, ".write(");
    p = stmt_print_local_index(p, stmt);    
  }
  p = isl_printer_print_str(p, ";");
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
	struct polysa_kernel_stmt *stmt;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	isl_ast_print_options_free(print_options);

	switch (stmt->type) {
	case POLYSA_KERNEL_STMT_COPY:
		return polysa_kernel_print_copy(p, stmt); 
	case POLYSA_KERNEL_STMT_SYNC: 
		return print_sync(p, stmt);
	case POLYSA_KERNEL_STMT_DOMAIN:
		return polysa_kernel_print_domain(p, stmt); 
	}

	return p;
}

static __isl_give isl_printer *print_module_stmt(__isl_take isl_printer *p,
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
    case POLYSA_KERNEL_STMT_SYNC:
      return print_sync(p, stmt);
    case POLYSA_KERNEL_STMT_DOMAIN:
      return polysa_kernel_print_domain(p, stmt);
    case POLYSA_KERNEL_STMT_IO:
      return polysa_kernel_print_io(p, stmt);
  }

  return p;
}

static void print_kernel(struct polysa_prog *prog, struct polysa_kernel *kernel,
	struct hls_info *hls)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	isl_ast_print_options *print_options;
	isl_printer *p;

	print_kernel_headers(prog, kernel, hls); 
	fprintf(hls->kernel_c, "{\n");
  /* Assign loop iterators to cuda identifiers. */
	// print_kernel_iterators(hls->kernel_c, kernel);

	p = isl_printer_to_file(ctx, hls->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_indent(p, 4);

//	p = print_kernel_vars(p, kernel);
	p = isl_printer_end_line(p);
	p = ppcg_set_macro_names(p);
	p = polysa_print_macros(p, kernel->tree);

//  // debug
//  isl_printer *p_d = isl_printer_to_file(isl_ast_node_get_ctx(kernel->tree), stdout);
//  p_d = isl_printer_set_output_format(p_d, ISL_FORMAT_C);
//  p_d = isl_printer_print_ast_node(p_d, kernel->tree);
//  printf("\n");
//  p_d = isl_printer_free(p_d);
//  // debug

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
						    &print_kernel_stmt, NULL); 
	p = isl_ast_node_print(kernel->tree, p, print_options);
	isl_printer_free(p);

	fprintf(hls->kernel_c, "}\n");
}

static void print_module(struct polysa_prog *prog, struct polysa_hw_module *module,
  struct hls_info *hls)
{
  isl_ctx *ctx = isl_ast_node_get_ctx(module->device_tree);
  isl_ast_print_options *print_options;
  isl_printer *p;

  /* TODO: print module headers */
//  print_module_headers(prog, module, hls);
  fprintf(hls->kernel_c, "{\n");

  p = isl_printer_to_file(ctx, hls->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_indent(p, 4);

  /* TODO */
//  p = print_module_vars(p, kernel);
  p = isl_printer_end_line(p);
 
//  // debug
//  isl_printer *p_d = isl_printer_to_file(isl_ast_node_get_ctx(module->device_tree), stdout);
//  p_d = isl_printer_set_output_format(p_d, ISL_FORMAT_C);
//  p_d = isl_printer_print_ast_node(p_d, module->device_tree);
//  printf("\n");
//  p_d = isl_printer_free(p_d);
//  // debug

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
						    &print_module_stmt, NULL); // TODO 
	p = isl_ast_node_print(module->device_tree, p, print_options);
	isl_printer_free(p);

	fprintf(hls->kernel_c, "}\n");
}

/* TODO */
static __isl_give isl_printer *print_hw_module(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  int is_user;
  struct polysa_hw_module *module;
  struct print_hw_module_data *data;

  isl_ast_print_options_free(print_options);
  
  data = (struct print_hw_module_data *) user;

  id = isl_ast_node_get_annotation(node);
  if (!id)
    return p;

  is_user = !strcmp(isl_id_get_name(id), "user");
  module = is_user ? NULL : isl_id_get_user(id);
  isl_id_free(id);

  if (is_user)
    return p;

  print_module(data->prog, module, data->hls);

  return p;
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
    return print_device_node(p, node, data->prog); 

  is_user = !strcmp(isl_id_get_name(id), "user");
  kernel = is_user ? NULL : isl_id_get_user(id);
  stmt = is_user ? isl_id_get_user(id) : NULL;
  isl_id_free(id);

  if (is_user)
    return polysa_kernel_print_domain(p, stmt); 

  p = ppcg_start_block(p); 

  // TODO: CUDA host kernel launch, to be modified later
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "dim3 k");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "_dimBlock");
  print_reverse_list(isl_printer_get_file(p),
      kernel->n_block, kernel->block_dim);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  p = print_grid(p, kernel); 

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "kernel");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, " <<<k");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "_dimGrid, k");
  p = isl_printer_print_int(p, kernel->id);
  p = isl_printer_print_str(p, "_dimBlock>>> (");
  p = print_kernel_arguments(p, data->prog, kernel, 0); 
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "cudaCheckKernel();");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  print_kernel(data->prog, kernel, data->hls); 

  return p;
}

static __isl_give isl_printer *print_host_code(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct hls_info *hls)
{
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_ast_node_get_ctx(tree);
  struct print_host_user_data data = { hls, prog };
  struct print_hw_module_data hw_data = { hls, prog };

  /* Print the default AST. */
  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                &print_host_user, &data); 

  /* Print the macros definitions in the program. */
  p = polysa_print_macros(p, tree); 
  p = isl_ast_node_print(tree, p, print_options);

  /* Print the hw module ASTs. */
  for (int i = 0; i < n_modules; i++) {
//  for (int i = 0; i < 1; i++) {
    print_options = isl_ast_print_options_alloc(ctx);
    print_options = isl_ast_print_options_set_print_user(print_options,
                  &print_hw_module, &hw_data);

    p = isl_ast_node_print(modules[i]->tree, p, print_options);
  }

  return p;
}

/* Given a polysa_prog "prog" and the corresponding tranformed AST
 * "tree", print the entire HLS code to "p".
 * "types" collecs the types for which a definition has already been
 * printed.
 */
static __isl_give isl_printer *print_hls(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct polysa_types *types, void *user)
{
  struct hls_info *hls = user;
  isl_printer *kernel;

  kernel = isl_printer_to_file(isl_printer_get_ctx(p), hls->kernel_c);
  kernel = isl_printer_set_output_format(kernel, ISL_FORMAT_C);
  kernel = polysa_print_types(kernel, types, prog);
  isl_printer_free(kernel);

  if (!kernel)
    return isl_printer_free(p);

  p = print_host_code(p, prog, tree, modules, n_modules, hls); 

  return p;
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
int generate_polysa_xilinx_hls(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input) 
{
  struct hls_info hls;
  int r;

  hls_open_files(&hls, input);

  r = generate_sa(ctx, input, hls.host_c, options, &print_hls, &hls);

  hls_close_files(&hls);
}
