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
#include "polysa_fpga.h"

#define min(a, b) ((a) < (b)? (a) : (b))

struct print_host_user_data {
	struct hls_info *hls;
	struct polysa_prog *prog;
  struct polysa_hw_top_module *top;
};

struct print_hw_module_data {
  struct hls_info *hls;
  struct polysa_prog *prog;
  struct polysa_hw_module *module;
};

/* Open the host .c file and the kernel .h and .c files for writing.
 * Add the necessary includes.
 */
static void hls_open_files(struct hls_info *info, const char *input)
{
  char name[PATH_MAX];
  int len;

  len = ppcg_extract_base_name(name, input);

  strcpy(name + len, "_host.cpp");
  info->host_c = fopen(name, "w");

  strcpy(name + len, "_kernel.cpp");
  info->kernel_c = fopen(name, "w");

  strcpy(name + len, "_kernel.h");
  info->kernel_h = fopen(name, "w");

  fprintf(info->host_c, "#include <assert.h>\n");
  fprintf(info->host_c, "#include <stdio.h>\n");
  fprintf(info->host_c, "#include \"xcl2.hpp\"\n");
  fprintf(info->host_c, "#include <algorithm>\n");
  fprintf(info->host_c, "#include <vector>\n");
  fprintf(info->host_c, "#include \"%s\"\n\n", name); 

  fprintf(info->kernel_c, "#include \"%s\"\n", name);  

  strcpy(name + len, "_top_gen.cpp");
  info->top_gen_c = fopen(name, "w");

  strcpy(name + len, "_top_gen.h");
  info->top_gen_h = fopen(name, "w");

  fprintf(info->host_c, "#include \"%s\"\n", name);  
  fprintf(info->top_gen_c, "#include <isl/printer.h>\n");
  fprintf(info->top_gen_c, "#include \"%s\"\n", name);  

  fprintf(info->kernel_h, "#include <ap_int.h>\n");
  fprintf(info->kernel_h, "#include <hls_stream.h>\n");
  fprintf(info->kernel_h, "\n");

  fprintf(info->kernel_h, "template<typename To, typename From>\n");
  fprintf(info->kernel_h, "inline To Reinterpret(const From &val) {\n");
  fprintf(info->kernel_h, "  return reinterpret_cast<const To &>(val);\n");
  fprintf(info->kernel_h, "}\n");
  fprintf(info->kernel_h, "\n");
}

static void opencl_open_files(struct hls_info *info, const char *input)
{
  char name[PATH_MAX];
  int len;

  len = ppcg_extract_base_name(name, input);

  strcpy(name + len, "_host.cpp");
  info->host_c = fopen(name, "w");

  strcpy(name + len, "_kernel.c");
  info->kernel_c = fopen(name, "w");

  strcpy(name + len, "_kernel.h");
  info->kernel_h = fopen(name, "w");

  fprintf(info->host_c, "#include <assert.h>\n");
  fprintf(info->host_c, "#include <stdio.h>\n");
  fprintf(info->host_c, "#include <math.h>\n");
  fprintf(info->host_c, "#include <CL/opencl.h>\n");
  fprintf(info->host_c, "#include \"AOCLUtils/aocl_utils.h\"\n");
  fprintf(info->host_c, "#include \"%s\"\n", name); 
  fprintf(info->host_c, "using namespace aocl_utils;\n\n");
  fprintf(info->host_c, "#define AOCX_FIEL \"krnl.aocx\"\n\n");

  /* Print Intel helper function */
  fprintf(info->host_c, "#define HOST\n");
  fprintf(info->host_c, "#define ACL_ALIGNMENT 64\n");
  fprintf(info->host_c, "#ifdef _WIN32\n");
  fprintf(info->host_c, "void *acl_aligned_malloc(size_t size) {\n");
  fprintf(info->host_c, "    return _aligned_malloc(size, ACL_ALIGNMENT);\n");
  fprintf(info->host_c, "}\n");
  fprintf(info->host_c, "void acl_aligned_free(void *ptr) {\n");
  fprintf(info->host_c, "    _aligned_free(ptr);\n");
  fprintf(info->host_c, "}\n");
  fprintf(info->host_c, "#else\n");
  fprintf(info->host_c, "void *acl_aligned_malloc(size_t size) {\n");
  fprintf(info->host_c, "    void *result = NULL;\n");
  fprintf(info->host_c, "    if (posix_memalign(&result, ACL_ALIGNMENT, size) != 0)\n");
  fprintf(info->host_c, "        printf(\"acl_aligned_malloc() failed.\\n\");\n");
  fprintf(info->host_c, "    return result;\n");
  fprintf(info->host_c, "}\n");
  fprintf(info->host_c, "void acl_aligned_free(void *ptr) {\n");
  fprintf(info->host_c, "    free(ptr);\n");
  fprintf(info->host_c, "}\n");
  fprintf(info->host_c, "#endif\n\n");

  fprintf(info->host_c, "void cleanup_host_side_resources();\n");
  fprintf(info->host_c, "void cleanup();\n\n");

  fprintf(info->host_c, "#define CHECK(status) \\\n");
  fprintf(info->host_c, "if (status != CL_SUCCESS) { \\\n");
  fprintf(info->host_c, "    fprintf(stderr, \"error %%d in line %%d.\\n\", status, __LINE__); \\\n");
  fprintf(info->host_c, "    exit(1); \\\n");
  fprintf(info->host_c, "}\n\n");

  fprintf(info->host_c, "#define CHECK_NO_EXIT(status) \\\n");
  fprintf(info->host_c, "if (status != CL_SUCCESS) { \\\n");
  fprintf(info->host_c, "    fprintf(stderr, \"error %%d in line %%d.\\n\", status, __LINE__); \\\n");
  fprintf(info->host_c, "}\n\n");

  fprintf(info->kernel_c, "#include \"%s\"\n", name);  

  strcpy(name + len, "_top_gen.c");
  info->top_gen_c = fopen(name, "w");

  strcpy(name + len, "_top_gen.h");
  info->top_gen_h = fopen(name, "w");

  fprintf(info->host_c, "#include \"%s\"\n", name);  
  fprintf(info->top_gen_c, "#include <isl/printer.h>\n");
  fprintf(info->top_gen_c, "#include \"%s\"\n", name);  
}

/* Close all output files.
 */
static void hls_close_files(struct hls_info *info)
{
  fclose(info->kernel_c);
  fclose(info->kernel_h);
  fclose(info->host_c);
  fclose(info->top_gen_c);
  fclose(info->top_gen_h);
//  fclose(info->top_gen_host_c);
}

/* Print the call of an array argument.
 */
static __isl_give isl_printer *polysa_array_info_print_call_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
	if (polysa_array_is_read_only_scalar(array))
		return isl_printer_print_str(p, array->name);

	p = isl_printer_print_str(p, "dev_");
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
__isl_give isl_printer *print_kernel_arguments(__isl_take isl_printer *p,
	struct polysa_prog *prog, struct polysa_kernel *kernel, int types)
{
	int i, n;
	int first = 1;
	unsigned nparam;
	isl_space *space;
	const char *type;

	for (i = 0; i < kernel->n_array; ++i) {
		int required;
    int n_lane;

		required = polysa_kernel_requires_array_argument(kernel, i);
		if (required < 0)
			return isl_printer_free(p);
		if (!required)
			continue;

    struct polysa_local_array_info *local_array = &kernel->array[i]; 
    n_lane = local_array->n_lane;

		if (!first)
			p = isl_printer_print_str(p, ", ");

		if (types)
			p = polysa_array_info_print_declaration_argument(p,
				local_array->array, n_lane, NULL);
		else
			p = polysa_array_info_print_call_argument(p,
				local_array->array);

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

static __isl_give isl_printer *print_set_kernel_arguments_xilinx(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_kernel *kernel)
{
  int n_arg = 0, n;
  unsigned nparam;
  isl_space *space;
  const char *type;

  /* array */
  for (int i = 0; i < prog->n_array; ++i) {
    int required;

    required = polysa_kernel_requires_array_argument(kernel, i);
    if (required < 0)
      return isl_printer_free(p);
    if (!required)
      continue;

    struct polysa_array_info *array = &prog->array[i];

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "OCL_CHECK(err, err = krnl.setArg(");
    p = isl_printer_print_int(p, n_arg);
    p = isl_printer_print_str(p, ", buffer_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, "));");
    p = isl_printer_end_line(p);
    n_arg++;
  }

  /* param */
	space = isl_union_set_get_space(kernel->arrays);
	nparam = isl_space_dim(space, isl_dim_param);
	for (int i = 0; i < nparam; ++i) {
		const char *name;
		name = isl_space_get_dim_name(space, isl_dim_param, i);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "OCL_CHECK(err, err = krnl.setArg(");
    p = isl_printer_print_int(p, n_arg);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, "));");
    p = isl_printer_end_line(p);
    n_arg++;
	}
	isl_space_free(space);

  /* host iterator */
	n = isl_space_dim(kernel->space, isl_dim_set);
	type = isl_options_get_ast_iterator_type(prog->ctx);
	for (int i = 0; i < n; ++i) {
		const char *name;
		name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "OCL_CHECK(err, err = krnl.setArg(");
    p = isl_printer_print_int(p, n_arg);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, "));");
    p = isl_printer_end_line(p);
    n_arg++;
	}

  return p;
}

static __isl_give isl_printer *print_top_gen_arguments(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_kernel *kernel, int types)
{
  int i, n;
  int first = 1;
  unsigned nparam;
  isl_space *space;
  const char *type;

  /* Parameters */
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

  /* Host iterators */
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

  /* File description */
  if (!first)
    p = isl_printer_print_str(p, ", ");
  if (types) {
    p = isl_printer_print_str(p, "FILE *");
  }
  p = isl_printer_print_str(p, "f");

  first = 0;

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

static __isl_give isl_printer *print_str_new_line(__isl_take isl_printer *p, const char *str) 
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, str);
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *declare_and_allocate_device_arrays_intel(__isl_take isl_printer *p, struct polysa_prog *prog)
{
  int indent;
  p = print_str_new_line(p, "// Allocate memory in host memory");
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, array->type);
    p = isl_printer_print_str(p, " *dev_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, " = (");
    p = isl_printer_print_str(p, array->type);
    p = isl_printer_print_str(p, "*)acl_aligned_malloc(");
    p = polysa_array_info_print_size(p, array);
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);   
  }
  p = isl_printer_end_line(p);

  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "memcpy(dev_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, ", ");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, ", ");
    p = polysa_array_info_print_size(p, array);
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "// Create device buffers");
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "buf_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, " = clCreateBuffer(context");
    p = isl_printer_end_line(p);
    indent = strlen("buf_") + strlen(array->name) + strlen(" clCreateBuffer(");
    p = isl_printer_indent(p, indent);
    p = print_str_new_line(p, "CL_MEM_READ_WRITE,");
    p = isl_printer_start_line(p);
    p = polysa_array_info_print_size(p, array);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
    p = print_str_new_line(p, "NULL,");
    p = print_str_new_line(p, "&status); CHECK(status);");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, -indent);
  }

  return p;
}

static __isl_give isl_printer *declare_and_allocate_device_arrays_xilinx(__isl_take isl_printer *p, struct polysa_prog *prog)
{
  p = print_str_new_line(p, "// Allocate Memory in Host Memory");
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "std::vector<");
    p = isl_printer_print_str(p, array->type);
    p = isl_printer_print_str(p, ", aligned_allocator<");
    p = isl_printer_print_str(p, array->type);
    p = isl_printer_print_str(p, ">> ");
    p = isl_printer_print_str(p, "dev_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, "(");
    p = polysa_array_info_print_data_size(p, array);
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "// Allocate Buffer in Global Memory");
  p = print_str_new_line(p, "// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and");
  p = print_str_new_line(p, "// Device-to-host communication");
  for (int i = 0; i < prog->n_array; i++) {
    int indent1, indent2;
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = print_str_new_line(p, "OCL_CHECK(err,");
    indent1 = strlen("OCL_CHECK(");
    p = isl_printer_indent(p, indent1);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "cl::Buffer buffer_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, "(context,");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, strlen("cl::Buffer buffer_") + strlen(array->name) + 1);
    p = print_str_new_line(p, "CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,");
    p = isl_printer_start_line(p);
    p = polysa_array_info_print_size(p, array);
    p = isl_printer_print_str(p, ",");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "dev_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, ".data(),");
    p = isl_printer_end_line(p);
    p = print_str_new_line(p, "&err));");
    p = isl_printer_indent(p, -(strlen("cl::Buffer buffer_") + strlen(array->name) + 1));
    p = isl_printer_indent(p, -indent1);
  }
  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);
    
  return p;
}

static __isl_give isl_printer *find_device_xilinx(__isl_take isl_printer *p)
{
  p = print_str_new_line(p, "if (argc != 2) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "std::cout << \"Usage: \" << argv[0] << \" <XCLBIN File>\" << std::endl;");
  p = print_str_new_line(p, "return EXIT_FAILURE;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "std::string binaryFile = argv[1];");
  p = print_str_new_line(p, "cl_int err;");
  p = print_str_new_line(p, "cl::Context context;");
  p = print_str_new_line(p, "cl::Kernel krnl;");
  p = print_str_new_line(p, "cl::CommandQueue q;");
  p = print_str_new_line(p, "// get_xil_devices() is a utility API which will find the xilinx");
  p = print_str_new_line(p, "// platforms and will return list of devices connected to Xilinx platform");
  p = print_str_new_line(p, "auto devices = xcl::get_xil_devices();");
  p = print_str_new_line(p, "// read_binary_file() is a utility API which will load the binaryFile");
  p = print_str_new_line(p, "// and will return the pointer to file buffer");
  p = print_str_new_line(p, "auto fileBuf = xcl::read_binary_file(binaryFile);");
  p = print_str_new_line(p, "cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};");
  p = print_str_new_line(p, "int valid_device = 0;");
  p = print_str_new_line(p, "for (unsigned int i = 0; i < devices.size(); i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "auto device = devices[i];");
  p = print_str_new_line(p, "// Creating Context and Command Queue for selected Device");
  p = print_str_new_line(p, "OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));");
  p = print_str_new_line(p, "OCL_CHECK(err, q = cl::CommandQueue(context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));");
  p = print_str_new_line(p, "std::cout << \"Trying to program device[\" << i");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "<< \"]: \" << device.getInfo<CL_DEVICE_NAME>() << std::endl;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, \%err));");
  p = print_str_new_line(p, "if (err != CL_SUCCESS) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "std::cout << \"Failed to program device[\" << i << \"] with xclbin file!\\n\";");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "} else {");
  p = isl_printer_indent(p, 4);  
  p = print_str_new_line(p, "std::cout << \"Device[\" << i << \"]: program successful!\\n\";");  
  p = print_str_new_line(p, "OCL_CHECK(err, krnl = cl::Kernel(program, \"kernel0\", &err));");
  p = print_str_new_line(p, "valid_device++");
  p = print_str_new_line(p, "break; // we break because we found a valid device");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "if (valid_device == 0) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "std::cout << \"Failed to program any device found, exit!\\n\";");
  p = print_str_new_line(p, "exit(EXIT_FAILURE);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);
}

static __isl_give isl_printer *find_device_intel(__isl_take isl_printer *p, struct polysa_prog *prog, struct polysa_hw_top_module *top)
{
  int indent;
  int n_cmd_q = 0;
  int n_kernel = 0;
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    n_cmd_q++;
  }

  for (int i = 0; i < top->n_hw_modules; i++) {
    struct polysa_hw_module *module = top->hw_modules[i];
    if (module->type != PE_MODULE && module->to_mem) {
      n_kernel++;
    }
  }

  p = print_str_new_line(p, "bool use_emulator = false; // control whether the emulator should be used.");
  p = print_str_new_line(p, "cl_int status;");
  p = print_str_new_line(p, "cl_platform_id platform = NULL;");
  p = print_str_new_line(p, "cl_device_id *devices = NULL;");
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "int NUM_QUEUES_TO_CREATE = ");
  p = isl_printer_print_int(p, n_cmd_q);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "int NUM_KERNELS_TO_CREATE = ");
  p = isl_printer_print_int(p, n_kernel);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "cl_kernel kernel[NUM_KERNELS_TO_CREATE];"); 
  p = print_str_new_line(p, "cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE];"); 
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "cl_mem buf_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, " = NULL;");
    p = isl_printer_end_line(p);
  }

  /* For each global array, we create one command queue */
  int q_id = 0;
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(array))
      continue;

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "int QID_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, " = ");
    p = isl_printer_print_int(p, q_id);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);
    q_id++;
  }
  /* For each I/O group, we create one kernel */
  int k_id = 0;
  for (int i = 0; i < top->n_hw_modules; i++) {
    struct polysa_hw_module *module = top->hw_modules[i];
    struct polysa_array_ref_group *group = module->io_groups[0];
    if (module->type != PE_MODULE && module->to_mem) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "int KID_");
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
      p = isl_printer_print_str(p, " = ");
      p = isl_printer_print_int(p, k_id);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
      k_id++;
    }
  }

  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "// Parse command line arguments");
  p = print_str_new_line(p, "Options options(argc, argv);");
  p = print_str_new_line(p, "if (options.has(\"emulator\")) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "use_emulator = options.get<bool>(\"emulator\")");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "if (!setCwdToExeDir()) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "return false");
  p = isl_printer_indent(p, -4);
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "// Get the OpenCL platform");
  p = print_str_new_line(p, "if (use_emulator) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "platform = findPlatform(\"Intel(R) FPGA Emulation Platform for OpenCL(TM)\");");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "} else {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "platform = findPlatform(\"Intel(R) FPGA SDK for OpenCL(TM)\");");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "if (platform == NULL) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "printf(\"ERROR: Unable to find Intel(R) FPGA OpenCL platform\");");
  p = print_str_new_line(p, "return -1;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "// Discover and initialize the devices");
  p = print_str_new_line(p, "cl_uint numDevices = 0;");
  p = print_str_new_line(p, "char buffer[4096];");
  p = print_str_new_line(p, "unsigned int buf_uint;");
  p = print_str_new_line(p, "int device_found = 0;");
  p = print_str_new_line(p, "status = clGetDeviceIDs(platform,");
  p = isl_printer_indent(p, strlen("status = clGetDeviceIDs("));
  p = print_str_new_line(p, "CL_DEVICE_TYPE_ALL,");
  p = print_str_new_line(p, "0,");
  p = print_str_new_line(p, "NULL,");
  p = print_str_new_line(p, "&numDevices);");
  indent = strlen("status = clGetDeviceIDs(");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "if (status == CL_SUCCESS) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "clGetPlatformInfo(platform,");
  p = isl_printer_indent(p, strlen("clGetPlatformInfo("));
  p = print_str_new_line(p, "CL_PLATFORM_VENDOR,");
  p = print_str_new_line(p, "4096,");
  p = print_str_new_line(p, "buffer,");
  p = print_str_new_line(p, "NULL);");
  indent = strlen("clGetPlatformInfo(");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "if (strstr(buffer, \"Intel(R)\") != NULL) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "device_found = 1;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "if (device_found) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "devices = (cl_device_id*) acl_aligned_malloc(numDevices * sizeof(cl_device_id));"); 
  p = print_str_new_line(p, "status = clGetDeviceIDs(platform,");
  p = isl_printer_indent(p, strlen("status = clGetDeviceIDs("));
  p = print_str_new_line(p, "CL_DEVICE_TYPE_ALL,");
  p = print_str_new_line(p, "numDevices,");
  p = print_str_new_line(p, "devices,");
  p = print_str_new_line(p, "NULL);");
  indent = strlen("status = clGetDeviceIDs(");
  p = isl_printer_indent(p, -indent);
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "if (!device_found) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "printf(\"failed to find a OpenCL device\\n\");");
  p = print_str_new_line(p, "exit(1);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");

  p = print_str_new_line(p, "for (int i = 0; i < numDevices; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "clGetDeviceInfo(devices[i],");
  indent = strlen("clGetDeviceInfo(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "CL_DEVICE_NAME,");
  p = print_str_new_line(p, "4096,");
  p = print_str_new_line(p, "buffer,");
  p = print_str_new_line(p, "NULL);");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "fprintf(stdout, \"\\nDevice Name: %s\\n\", buffer);");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "clGetDeviceInfo(devices[i],");
  indent = strlen("clGetDeviceInfo(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "CL_DEVICE_VENDOR,");
  p = print_str_new_line(p, "4096,");
  p = print_str_new_line(p, "buffer,");
  p = print_str_new_line(p, "NULL);");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "fprintf(stdout, \"Device Vendor: %s\\n\", buffer);");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "clGetDeviceInfo(devices[i],");
  indent = strlen("clGetDeviceInfo(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "CL_DEVICE_MAX_COMPUTE_UNITS,");
  p = print_str_new_line(p, "sizeof(buf_uint),");
  p = print_str_new_line(p, "&buf_uint,");
  p = print_str_new_line(p, "NULL);");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "fprintf(stdout, \"Device Computing Units: %u\\n\", buf_uint);");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "clGetDeviceInfo(devices[i],");
  indent = strlen("clGetDeviceInfo(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "CL_DEVICE_GLOBAL_MEM_SIZE,");
  p = print_str_new_line(p, "sizeof(unsigned long),");
  p = print_str_new_line(p, "&buffer,");
  p = print_str_new_line(p, "NULL);");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "fprintf(stdout, \"Global Memory Size: %lu\\n\", *((unsigned long*)buffer));");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "clGetDeviceInfo(devices[i],");
  indent = strlen("clGetDeviceInfo(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "CL_DEVICE_MAX_MEM_ALLOC_SIZE,");
  p = print_str_new_line(p, "sizeof(unsigned long),");
  p = print_str_new_line(p, "&buffer,");
  p = print_str_new_line(p, "NULL);");
  p = isl_printer_indent(p, -indent);
  p = print_str_new_line(p, "fprintf(stdout, \"Global Memory Allocation Size: %lu\\n\\n\", *((unsigned long*)buffer));");
  p = isl_printer_end_line(p);

  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);

  /* Context */
  p = print_str_new_line(p, "// Create a context");
  p = print_str_new_line(p, "context = clCreateContext(NULL,");
  indent = strlen("context = clCreateContext(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "1,");
  p = print_str_new_line(p, "devices,");
  p = print_str_new_line(p, "NULL,");
  p = print_str_new_line(p, "NULL,");
  p = print_str_new_line(p, "&status); CHECK(status);");
  p = isl_printer_indent(p, -indent);
  p = isl_printer_end_line(p);

  /* Command Queue */
  p = print_str_new_line(p, "// Create command queues");
  p = print_str_new_line(p, "for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "cmdQueue[i] = clCreateCommandQueue(context");
  indent = strlen("cmdQueue[i] = clCreateCommandQueue(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "devices[0],");
  p = print_str_new_line(p, "CL_QUEUE_PROFILING_ENABLE,");
  p = print_str_new_line(p, "&status); CHECK(status);");
  p = isl_printer_indent(p, -indent);
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);  

  /* Create the program from binaries */
  p = print_str_new_line(p, "// Create the program from binaries");
  p = print_str_new_line(p, "size_t binary_length;");
  p = print_str_new_line(p, "const unsigned char *binary;");
  p = print_str_new_line(p, "printf(\"\\nAOCX file: %%s\\n\\n\", AOCX_FILE);");
  p = print_str_new_line(p, "FILE *fp = fopen(AOCX_FILE, \"rb\");");
  p = print_str_new_line(p, "if (fp == NULL) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "printf(\"Failed to open the AOCX file (fopen).\\n\");");
  p = print_str_new_line(p, "return -1;");  
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "fseek(fp, 0, SEEK_END);");
  p = print_str_new_line(p, "long ftell_sz = ftell(fp);");
  p = print_str_new_line(p, "if (ftell_sz < 0) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "printf(\"ftell returns a negative value.\\n\");");
  p = print_str_new_line(p, "fclose(fp);");
  p = print_str_new_line(p, "return -1;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "} else {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "binary_length = ftell_sz;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "binary = (unsigned char *)malloc(sizeof(unsigned char) * binary_length);");
  p = print_str_new_line(p, "rewind(fp);");
  p = print_str_new_line(p, "size_t fread_sz = fread((void *)binary, binary_length, 1, fp);");
  p = print_str_new_line(p, "if (fread_sz == 0) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "printf(\"Failed to read from the AOCX file (fread).\\n\");");
  p = print_str_new_line(p, "fclose(fp);");
  p = print_str_new_line(p, "free(const_char<unsigned char *>(binary))");
  p = print_str_new_line(p, "return -1;");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = print_str_new_line(p, "fclose(fp);");
  p = isl_printer_end_line(p);
  
  p = print_str_new_line(p, "program = clCreateProgramWithBinary(context,");
  indent = strlen("program = clCreateProgramWithBinary(");
  p = isl_printer_indent(p, indent);
  p = print_str_new_line(p, "1,");
  p = print_str_new_line(p, "devices,");
  p = print_str_new_line(p, "&binary_length,");
  p = print_str_new_line(p, "(const unsigned char **)&binary,");
  p = print_str_new_line(p, "&status,");
  p = print_str_new_line(p, "NULL); CHECK(status);");
  p = isl_printer_indent(p, -indent);
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);");
  p = print_str_new_line(p, "if (status != CL_SUCCESS) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "char log[10000] = {0};");
  p = print_str_new_line(p, "clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 10000, log, NULL);");
  p = print_str_new_line(p, "printf(\"%%s\\n\", log);");
  p = print_str_new_line(p, "CHECK(status);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);

  /* Create the kernel */
  p = print_str_new_line(p, "// Create the kernel");
  p = print_str_new_line(p, "for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "kernel[i] = clCreateKernel(program, NULL, &status);");
  p = print_str_new_line(p, "CHECK(status);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");

  return p;
}

/* Print code for initializing the device for execution of the transformed
 * code.  This includes declaring locally defined variables as well as
 * declaring and allocating the required copies of arrays on the device.
 */
static __isl_give isl_printer *init_device_xilinx(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
	p = polysa_print_local_declarations(p, prog);
  p = find_device_xilinx(p);
  p = declare_and_allocate_device_arrays_xilinx(p, prog); 
//	p = declare_device_arrays(p, prog);
//	p = allocate_device_arrays(p, prog);

	return p;
}

/* Print code for initializing the device for execution of the transformed
 * code.  This includes declaring locally defined variables as well as
 * declaring and allocating the required copies of arrays on the device.
 */
static __isl_give isl_printer *init_device_intel(__isl_take isl_printer *p,
	struct polysa_prog *prog, struct polysa_hw_top_module *top)
{
	p = polysa_print_local_declarations(p, prog);
  p = find_device_intel(p, prog, top);
  p = declare_and_allocate_device_arrays_intel(p, prog);
//	p = declare_device_arrays(p, prog);
//	p = allocate_device_arrays(p, prog);

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
static __isl_give isl_printer *clear_device_intel(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
//	p = free_device_arrays(p, prog);
//  p = print_str_new_line(p, "cleanup();");
  p = print_str_new_line(p, "// clean up resources");
  for (int i = 0; i < prog->n_array; i++) {
    struct polysa_array_info *array = &prog->array[i];
    if (!polysa_array_requires_device_allocation(&prog->array[i])) 
      continue;
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "acl_aligned_free(dev_");
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "clReleaseKernel(kernel[i]);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "clReleaseCommandQueue(cmdQueue[i]);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "clReleaseProgram(program);");
  p = print_str_new_line(p, "clReleaseContext(context);");
  p = print_str_new_line(p, "acl_aligned_free(devices);");

	return p;
}

/* Print code for clearing the device after execution of the transformed code.
 * In particular, free the memory that was allocated on the device.
 */
static __isl_give isl_printer *clear_device_xilinx(__isl_take isl_printer *p,
	struct polysa_prog *prog)
{
//	p = free_device_arrays(p, prog);
  p = print_str_new_line(p, "q.finish();");

	return p;
}

/* Print code to "p" for copying "array" from the host to the device
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_to_device_intel(__isl_take isl_printer *p,
	struct polysa_array_info *array)
{
  int indent;
  p = print_str_new_line(p, "status = clEnqueueWriteBuffer(");
  p = isl_printer_indent(p, 4);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "cmdQueue[QID_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, "],");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "buf_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, ",");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "CL_TRUE,");
  p = print_str_new_line(p, "0,");
  p = isl_printer_start_line(p);
  p = polysa_array_info_print_size(p, array);
  p = isl_printer_print_str(p, ",");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "dev_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, ",");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "0,");
  p = print_str_new_line(p, "NULL,");
  p = print_str_new_line(p, "NULL); CHECK(status);");
  p = isl_printer_indent(p, -4);
  p = isl_printer_end_line(p);

	return p;
}

/* Print code to "p" for copying "array" back from the device to the host
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * polysa_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_from_device_intel(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
  p = print_str_new_line(p, "status = clEnqueueReadBuffer(");
  p = isl_printer_indent(p, 4);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "cmdQueue[QID_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, "],");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "buf_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, ",");
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "CL_TRUE,");
  p = print_str_new_line(p, "0,");
  p = isl_printer_start_line(p);
  p = polysa_array_info_print_size(p, array);
  p = isl_printer_print_str(p, ",");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "dev_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_end_line(p);
  p = print_str_new_line(p, "0,");
  p = print_str_new_line(p, "NULL,");
  p = print_str_new_line(p, "NULL); CHECK(status);");
  p = isl_printer_indent(p, -4);
  p = isl_printer_end_line(p);

	return p;
}

/* Print code to "p" for copying "array" from the host to the device
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_to_device_xilinx(__isl_take isl_printer *p,
	struct polysa_array_info *array)
{
  int indent;
  p = print_str_new_line(p, "OCL_CHECK(err,");
  indent = strlen("OCL_CHECK(");
  p = isl_printer_indent(p, indent);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "err = q.enqueueMigrateMemObjects({buffer_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, "}, 0));");
  p = isl_printer_end_line(p);
  p = isl_printer_indent(p, -indent);

	return p;
}

/* Print code to "p" for copying "array" back from the device to the host
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * polysa_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_from_device_xilinx(
	__isl_take isl_printer *p, struct polysa_array_info *array)
{
  int indent;
  p = print_str_new_line(p, "OCL_CHECK(err,");
  indent = strlen("OCL_CHECK(");
  p = isl_printer_indent(p, indent);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "err = q.enqueueMigrateMemObjects({buffer_");
  p = isl_printer_print_str(p, array->name);
  p = isl_printer_print_str(p, "}, CL_MIGRATE_MEM_OBJECT_HOST));");
  p = isl_printer_end_line(p);
  p = isl_printer_indent(p, -indent);

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
static __isl_give isl_printer *print_device_node_xilinx(__isl_take isl_printer *p,
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
		return init_device_xilinx(p, prog); 
	if (!strcmp(name, "clear_device"))
		return clear_device_xilinx(p, prog); 
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array_to_device_xilinx(p, array); 
	else
		return copy_array_from_device_xilinx(p, array); 
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
static __isl_give isl_printer *print_device_node_intel(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node, struct polysa_prog *prog, struct polysa_hw_top_module *top)
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
		return init_device_intel(p, prog, top); 
	if (!strcmp(name, "clear_device"))
		return clear_device_intel(p, prog); 
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array_to_device_intel(p, array); 
	else
		return copy_array_from_device_intel(p, array); 
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
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* CPU Kernel Definition */");
  p = isl_printer_end_line(p);
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

/* Print the header of the given kernel to both gen->hls.kernel_h
 * and gen->hls.kernel_c.
 */
static void print_kernel_headers_xilinx(struct polysa_prog *prog,
	struct polysa_kernel *kernel, struct hls_info *hls)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, hls->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_str_new_line(p, "extern \"C\" {");
	p = print_kernel_header(p, prog, kernel); 
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
  p = print_str_new_line(p, "}");
	isl_printer_free(p);
}

static void print_indent(FILE *dst, int indent)
{
	fprintf(dst, "%*s", indent, "");
}

/* Print a list of iterators of type "type" with names "ids" to "out".
 * Each iterator is assigned one of the instance identifiers in dims.
 */
static void print_iterators(FILE *out, const char *type,
	__isl_keep isl_id_list *ids, const char *dims[])
{
	int i, n;
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_id_list_get_ctx(ids), stdout);
//  // debug

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
//    // debug
//    p = isl_printer_print_id(p, id);
//    printf("\n");
//    // debug
		fprintf(out, "%s = %s", isl_id_get_name(id),
			dims[i]);
		isl_id_free(id);
	}
	fprintf(out, "; // module id\n");
}

static __isl_give isl_printer *print_kernel_var_xilinx(__isl_take isl_printer *p,
	struct polysa_kernel_var *var, int double_buffer)
{
	int j;

	p = isl_printer_start_line(p);
  if (var->n_lane == 1)
	  p = isl_printer_print_str(p, var->array->type);
  else {
    p = isl_printer_print_str(p, var->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, var->n_lane);
  }
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
  if (double_buffer)
    p = isl_printer_print_str(p, "_ping");
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
  if (var->n_part != 1) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS ARRAY_PARTITION variable=");
    p = isl_printer_print_str(p, var->name);
    p = isl_printer_print_str(p, " dim=");
    p = isl_printer_print_int(p, isl_vec_size(var->size));
    p = isl_printer_print_str(p, " factor=");
    p = isl_printer_print_int(p, var->n_part);
    p = isl_printer_print_str(p, " cyclic");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "#pragma HLS RESOURCE variable=");
  p = isl_printer_print_str(p, var->name);
  if (double_buffer)
    p = isl_printer_print_str(p, "_ping");
  p = isl_printer_print_str(p, " core=RAM_2P_BRAM");
  p = isl_printer_end_line(p);

  /* Print pong buffer */
  if (double_buffer) {
  	p = isl_printer_start_line(p);
    if (var->n_lane == 1)
      p = isl_printer_print_str(p, var->array->type);
    else {
      p = isl_printer_print_str(p, var->array->name);
      p = isl_printer_print_str(p, "_t");
      p = isl_printer_print_int(p, var->n_lane);
    }
  	p = isl_printer_print_str(p, " ");
  	p = isl_printer_print_str(p,  var->name);
    if (double_buffer)
      p = isl_printer_print_str(p, "_pong");
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
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS RESOURCE variable=");
    p = isl_printer_print_str(p, var->name);
    p = isl_printer_print_str(p, "_pong");
    p = isl_printer_print_str(p, " core=RAM_2P_BRAM");
    p = isl_printer_end_line(p);
  }

	return p;
}

static __isl_give isl_printer *print_kernel_var_intel(__isl_take isl_printer *p,
	struct polysa_kernel_var *var, int double_buffer)
{
	int j;

	p = isl_printer_start_line(p);
  if (var->n_lane == 1)
	  p = isl_printer_print_str(p, var->array->type);
  else {
    p = isl_printer_print_str(p, var->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, var->n_lane);
  }
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
  if (double_buffer)
    p = isl_printer_print_str(p, "_ping");
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

  /* Print pong buffer */
  if (double_buffer) {
  	p = isl_printer_start_line(p);
    if (var->n_lane == 1)
      p = isl_printer_print_str(p, var->array->type);
    else {
      p = isl_printer_print_str(p, var->array->name);
      p = isl_printer_print_str(p, "_t");
      p = isl_printer_print_int(p, var->n_lane);
    }
  	p = isl_printer_print_str(p, " ");
  	p = isl_printer_print_str(p,  var->name);
    if (double_buffer)
      p = isl_printer_print_str(p, "_pong");
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
  }

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

/* Print an access to the element in the global memory copy
 * described by "stmt".  The index of the copy is recorded in
 * stmt->index as an access to the array.
 */
static __isl_give isl_printer *io_stmt_print_global_index(
	__isl_take isl_printer *p, struct polysa_kernel_stmt *stmt)
{
	struct polysa_array_info *array = stmt->u.i.array;
	isl_ast_expr *index;

	if (polysa_array_is_scalar(array)) {
		if (!polysa_array_is_read_only_scalar(array))
			p = isl_printer_print_str(p, "*");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	index = isl_ast_expr_copy(stmt->u.i.index);

	p = isl_printer_print_ast_expr(p, index);
	isl_ast_expr_free(index);

	return p;
}

/* If read, print:
 *   "read_channel_intel([fifo_name])"
 * else, print:
 *   "write_channel_intel([fifo_name],"
 */
static __isl_give isl_printer *print_fifo_rw_intel(__isl_take isl_printer *p,
  const char *fifo_name, int read)
{
  if (read) {
    p = isl_printer_print_str(p, "read_channel_intel(");
    p = isl_printer_print_str(p, fifo_name);
    p = isl_printer_print_str(p, ")");
  } else {
    p = isl_printer_print_str(p, "write_channel_intel(");
    p = isl_printer_print_str(p, fifo_name);
    p = isl_printer_print_str(p, ", ");
  }

  return p;
}

/* Print out
 * "channel [type]"
 */
static __isl_give isl_printer *print_fifo_type_intel(__isl_take isl_printer *p, 
  struct polysa_array_ref_group *group, int n_lane)
{
  p = isl_printer_print_str(p, "channel ");
  if (n_lane == 1)
    p = isl_printer_print_str(p, group->array->type);
  else {
    p = isl_printer_print_str(p, group->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, n_lane);
  }

  return p;
}

/* If read, print:
 *   "[fifo_name].read()"
 * else, print:
 *   "[fifo_name].write("
 */
static __isl_give isl_printer *print_fifo_rw_xilinx(__isl_take isl_printer *p,
  const char *fifo_name, int read)
{
  if (read) {
    p = isl_printer_print_str(p, fifo_name);
    p = isl_printer_print_str(p, ".read()");    
  } else {
    p = isl_printer_print_str(p, fifo_name);
    p = isl_printer_print_str(p, ".write(");
  }
  return p;
}

/* Print out
 * "hls::stream<[type]>"
 */
static __isl_give isl_printer *print_fifo_type_xilinx(__isl_take isl_printer *p, 
  struct polysa_array_ref_group *group, int n_lane)
{
  p = isl_printer_print_str(p, "hls::stream<");
  if (n_lane == 1) {
    p = isl_printer_print_str(p, group->array->type);
  } else {
    struct polysa_array_info *array = group->array;
    p = isl_printer_print_str(p, array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, n_lane);
  }
  p = isl_printer_print_str(p, ">");
}

static __isl_give isl_printer *polysa_fifo_print_declaration_arguments(
  __isl_take isl_printer *p, struct polysa_array_ref_group *group, int n_lane,
  const char *suffix, enum platform target)
{
  if (target == XILINX_HW) {
    p = print_fifo_type_xilinx(p, group, n_lane);
    p = isl_printer_print_str(p, " &");
  } else {
    p = print_fifo_type_intel(p, group, n_lane);
    p = isl_printer_print_str(p, " ");
  }
  p = polysa_array_ref_group_print_fifo_name(group, p); 
  if (suffix) {
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_str(p, suffix);
  }
  
  return p;
}

static __isl_give isl_printer *polysa_fifo_print_call_argument(
  __isl_take isl_printer *p, struct polysa_array_ref_group *group,
  const char *suffix, enum platform target)
{
  p = polysa_array_ref_group_print_fifo_name(group, p);
  if (suffix) {
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_str(p, suffix);
  }

  return p;
}

/* Print the arguments to a module declaration or call. If "types" is set,
 * then print a declaration (including the types of the arguments).
 *
 * The arguments are printed in the following order
 * - the module identifiers
 * - the host loop iterators
 * - the parameters
 * - the arrays accessed by the module
 * - the fifos
 */
static __isl_give isl_printer *print_pe_dummy_module_arguments(
  __isl_take isl_printer *p,
  struct polysa_prog *prog,
  struct polysa_kernel *kernel,
  struct polysa_pe_dummy_module *pe_dummy_module, 
  int types,
  enum platform target)
{
  int first = 1;
  isl_space *space;
  int nparam;
  int n;
  const char *type;
  struct polysa_hw_module *module = pe_dummy_module->module;

  type = isl_options_get_ast_iterator_type(prog->ctx);
  /* module identifiers */
  const char *dims[] = { "idx", "idy", "idz" };
  n = isl_id_list_n_id(module->inst_ids);
  for (int i = 0; i < n; ++i) {
    if (!first)
      p = isl_printer_print_str(p, ", ");
    if (types) {
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
    }
    p = isl_printer_print_str(p, dims[i]);

    first = 0;
  }

  /* params */
  space = isl_union_set_get_space(kernel->arrays); 
  nparam = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < nparam; ++i) {
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

  /* host iters */
  space = module->space;

  n = isl_space_dim(space, isl_dim_set); 
  for (int i = 0; i < n; ++i) {
    const char *name;
  
    if (!first)
      p = isl_printer_print_str(p, ", ");
    name = isl_space_get_dim_name(space, isl_dim_set, i);
    if (types) {
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
    }
    p = isl_printer_print_str(p, name); 
    
    first = 0;
  }

  /* Arrays */
  /* Scalars */
  for (int i = 0; i < prog->n_array; i++) {
    int required;

    required = polysa_kernel_requires_array_argument(kernel, i);
    if (required < 0)
      return isl_printer_free(p);
    if (!required)
      continue;

    if (polysa_array_is_read_only_scalar(&prog->array[i])) {
      if (!first) { 
        p = isl_printer_print_str(p, ", ");
      }
      if (types)
        p = polysa_array_info_print_declaration_argument(p,
              &prog->array[i], 1, NULL);
      else
        p = polysa_array_info_print_call_argument(p,
              &prog->array[i]); 
      first = 0;
    }
  }

  /* fifos */
  {
    struct polysa_array_ref_group *group = pe_dummy_module->io_group;
    int n_lane = (group->local_array->array_type == POLYSA_EXT_ARRAY)? group->n_lane:
          ((group->group_type == POLYSA_DRAIN_GROUP)? group->n_lane: 
          ((group->io_type == POLYSA_EXT_IO)? group->n_lane : group->io_buffers[0]->n_lane));

    if (!first) {
      p = isl_printer_print_str(p, ", ");
    }
    if (types) {
      p = polysa_fifo_print_declaration_arguments(p,
            group, n_lane, "in", target); 
    } else 
      p = polysa_fifo_print_call_argument(p,
            group, "in", target); 
    first = 0;
  } 

  return p;
}

/* Print the arguments to a module declaration or call. If "types" is set,
 * then print a declaration (including the types of the arguments).
 *
 * The arguments are printed in the following order
 * - the module identifiers
 * - the host loop iterators
 * - the parameters
 * - the arrays accessed by the module
 * - the fifos
 * - the enable
 */
static __isl_give isl_printer *print_module_arguments(__isl_take isl_printer *p,
  struct polysa_prog *prog, 
  struct polysa_kernel *kernel,
  struct polysa_hw_module *module, int types,
  enum platform target,
  int inter, int arb, int boundary)
{
  int first = 1;
  isl_space *space;
  int nparam;
  int n;
  const char *type;

  type = isl_options_get_ast_iterator_type(prog->ctx);
  /* module identifiers */
  const char *dims[] = { "idx", "idy", "idz" };
  n = isl_id_list_n_id(module->inst_ids);
  for (int i = 0; i < n; ++i) {
    if (!first)
      p = isl_printer_print_str(p, ", ");
    if (types) {
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
    }
    p = isl_printer_print_str(p, dims[i]);

    first = 0;
  }

  /* params */
  space = isl_union_set_get_space(kernel->arrays); 
  nparam = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < nparam; ++i) {
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

  /* host iters */
  if (inter == -1)
    space = module->space;
  else if (inter == 0)
    space = module->intra_space;
  else if (inter == 1)
    space = module->inter_space;

  n = isl_space_dim(space, isl_dim_set); 
  for (int i = 0; i < n; ++i) {
    const char *name;
  
    if (!first)
      p = isl_printer_print_str(p, ", ");
    name = isl_space_get_dim_name(space, isl_dim_set, i);
    if (types) {
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
    }
    p = isl_printer_print_str(p, name);
    if (module->double_buffer && inter != -1) {
      if (module->in && inter == 0) {
        /* intra trans */
        p = isl_printer_print_str(p, "_prev");
      } else if (!module->in && inter == 1) {
        /* inter trans */
        p = isl_printer_print_str(p, "_prev");
      }
    }
    
    first = 0;
  }

  /* Arrays */
  if (module->type != PE_MODULE && module->to_mem) {
    /* I/O module that accesses the external memory */
    struct polysa_io_buffer *io_buffer = module->io_groups[0]->io_buffers[module->io_groups[0]->io_level - 1];
    int n_lane = io_buffer->n_lane; 
    if (!first) {
      p = isl_printer_print_str(p, ", ");
    }
    if (types) {
      p = polysa_array_info_print_declaration_argument(p,
            module->io_groups[0]->array, n_lane, target == INTEL_HW? "global" : NULL);
    } else {
      p = polysa_array_info_print_call_argument(p, 
            module->io_groups[0]->array); 
    }
    first = 0;
  } else if (module->type == PE_MODULE) {
    /* Scalars */
    for (int i = 0; i < prog->n_array; i++) {
      int required;

      required = polysa_kernel_requires_array_argument(kernel, i);
      if (required < 0)
        return isl_printer_free(p);
      if (!required)
        continue;

      if (polysa_array_is_read_only_scalar(&prog->array[i])) {
        if (!first) { 
          p = isl_printer_print_str(p, ", ");
        }
        if (types)
          p = polysa_array_info_print_declaration_argument(p,
                &prog->array[i], 1, NULL);
        else
          p = polysa_array_info_print_call_argument(p,
                &prog->array[i]); 
        first = 0;
      }
    }
  }

  /* Local buffer */
  if (inter != -1) {
    for (int i = 0; i < module->n_var; i++) {
      struct polysa_kernel_var *var;

      var = &module->var[i];
      if (!first)
        p = isl_printer_print_str(p, ", ");

      if (types) {
        if (module->data_pack_inter == 1) {
          p = isl_printer_print_str(p, var->array->type);
        } else {
          p = isl_printer_print_str(p, var->array->name);
          p = isl_printer_print_str(p, "_t");
          p = isl_printer_print_int(p, module->data_pack_inter);
        }
        p = isl_printer_print_str(p, " ");
        p = isl_printer_print_str(p, var->name);
        for (int j = 0; j < isl_vec_size(var->size); j++) {
          isl_val *v;

          p = isl_printer_print_str(p, "[");
          v = isl_vec_get_element_val(var->size, j);
          p = isl_printer_print_val(p, v);
          isl_val_free(v);
          p = isl_printer_print_str(p, "]");
        }
      } else {
        if (!module->double_buffer) {
          p = isl_printer_print_str(p, var->name);          
        } else {
          if (arb == 0) {
            p = isl_printer_print_str(p, var->name);
            p = isl_printer_print_str(p, inter == 0? "_ping" : "_pong");
          } else {
            p = isl_printer_print_str(p, var->name);
            p = isl_printer_print_str(p, inter == 0? "_pong" : "_ping");
          }
        }
      }

      first = 0;
    }
  }

  /* fifos */
  if (module->type == PE_MODULE) {
    for (int i = 0; i < module->n_io_group; i++) {
      struct polysa_array_ref_group *group = module->io_groups[i];
      int n_lane = (group->local_array->array_type == POLYSA_EXT_ARRAY)? group->n_lane:
          ((group->group_type == POLYSA_DRAIN_GROUP)? group->n_lane: 
          ((group->io_type == POLYSA_EXT_IO)? group->n_lane : group->io_buffers[0]->n_lane));

      if (module->io_groups[i]->pe_io_dir == IO_IN ||
          module->io_groups[i]->pe_io_dir == IO_INOUT) {
        if (!first) {
          p = isl_printer_print_str(p, ", ");
        }
        if (types) {
          p = polysa_fifo_print_declaration_arguments(p,
                module->io_groups[i], n_lane, "in", target); 
        } else 
          p = polysa_fifo_print_call_argument(p,
                module->io_groups[i], "in", target); 
        first = 0;
      } 
      if (module->io_groups[i]->pe_io_dir == IO_OUT ||
          module->io_groups[i]->pe_io_dir == IO_INOUT) {
        if (!first)
          p = isl_printer_print_str(p, ", ");
        if (types)
          p = polysa_fifo_print_declaration_arguments(p,
                module->io_groups[i], n_lane, "out", target);
        else
          p = polysa_fifo_print_call_argument(p,
                module->io_groups[i], "out", target);
        first = 0;
      }
    }
  } else {
    for (int i = 0; i < module->n_io_group; i++) {
      if (!module->to_mem && inter != 0) {
        if (!(!module->in && boundary)) {
          if (!first) {
            p = isl_printer_print_str(p, ", ");
          }
          /* in */
          if (types)
            p = polysa_fifo_print_declaration_arguments(p,
                  module->io_groups[i], module->data_pack_inter, "in", target); 
          else
            p = polysa_fifo_print_call_argument(p,
                  module->io_groups[i], "in", target);
          first = 0;
        }

        if (!(module->in && boundary)) {
          /* out */
          if (!first)
            p = isl_printer_print_str(p, ", ");
          if (types)
            p = polysa_fifo_print_declaration_arguments(p,
                  module->io_groups[i], module->data_pack_inter, "out", target); 
          else
            p = polysa_fifo_print_call_argument(p,
                  module->io_groups[i], "out", target);
          first = 0;
        }
      }

      if (inter != 1) {
        if (!first)
          p = isl_printer_print_str(p, ", ");
        /* local */
        if (types)
          p = polysa_fifo_print_declaration_arguments(p,
                module->io_groups[i], module->data_pack_intra, module->in? "local_out" : "local_in", target); 
        else
          p = polysa_fifo_print_call_argument(p,
                module->io_groups[i], module->in? "local_out" : "local_in", target);
        first = 0;
      }
    }
  }

  /* credit fifo */
  if (module->credit) {
    if (!first) {
      p = isl_printer_print_str(p, ", ");
    }
    if (types) {
      if (target == XILINX_HW) {
        p = isl_printer_print_str(p, "hls::stream<int> &credit");
      } else {
        p = isl_printer_print_str(p, "channel int credit");
      }
    } else {
      p = isl_printer_print_str(p, "credit");
    }
    
    first = 0;
  }

  /* enable */
  if (module->double_buffer && inter != -1) {
    if (!first) {
      p = isl_printer_print_str(p, ", ");
    }
    if (types) {
      p = isl_printer_print_str(p, inter == 0? "bool intra_trans_en" : "bool inter_trans_en");
    } else {
      p = isl_printer_print_str(p, inter == 0? "intra_trans_en" : "inter_trans_en");
    }

    first = 0;
  }

  return p;
}

/* Print an I/O statement.
 *
 * An in I/O statement is printed as 
 *
 *  [type] fifo_data;
 *  fifo_data = global;
 *  fifo.write(fifo_data);
 *
 * while an out I/O statement is printed as
 *
 *  [type] fifo_data;
 *  fifo_data = fifo.read();
 *  global = fifo_data;
 *
 */
__isl_give isl_printer *polysa_kernel_print_io_dram(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_array_ref_group *group = stmt->u.i.group;
  struct polysa_hw_module *module = stmt->u.i.module;
  char *fifo_name;
  int n_lane = stmt->u.i.data_pack;
  isl_ctx *ctx = isl_printer_get_ctx(p);
  int buf = stmt->u.i.buf;
  isl_ast_expr *local_index_packed;
  local_index_packed = isl_ast_expr_copy(stmt->u.i.local_index);
  int n_arg;
  /* Modify the local index */
  if (n_lane > 1) {
    isl_ast_expr *arg, *div;
    n_arg = isl_ast_expr_get_op_n_arg(local_index_packed);
    arg = isl_ast_expr_get_op_arg(local_index_packed, n_arg - 1);
    div = isl_ast_expr_from_val(isl_val_int_from_si(ctx, n_lane));
    arg = isl_ast_expr_div(arg, div);
    local_index_packed = isl_ast_expr_set_op_arg(local_index_packed, n_arg - 1, arg);
  }

  p = isl_printer_indent(p, -2);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "{");
  p = isl_printer_end_line(p);
  p = isl_printer_indent(p, 2);

  /* Declare the fifo data variable */
  p = isl_printer_start_line(p);
  if (n_lane == 1) {
    p = isl_printer_print_str(p, stmt->u.i.array->type);
  } else {
    p = isl_printer_print_str(p, stmt->u.i.array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, n_lane);
  }
  p = isl_printer_print_str(p, " fifo_data;");
  p = isl_printer_end_line(p);

  if (stmt->u.i.in) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_data = ");
    p = io_stmt_print_global_index(p, stmt);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);

    if (!buf) {
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "out");
      p = isl_printer_start_line(p);
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 0);
      else
        p = print_fifo_rw_intel(p, fifo_name, 0);
      p = isl_printer_print_str(p, "fifo_data);");
      p = isl_printer_end_line(p);
      free(fifo_name);
    } else {
      p = isl_printer_start_line(p);
      p = isl_printer_print_ast_expr(p, local_index_packed);
      p = isl_printer_print_str(p, " = fifo_data;");
      p = isl_printer_end_line(p);
    }
  } else {
    if (!buf) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "in");
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 1);
      else
        p = print_fifo_rw_intel(p, fifo_name, 1);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
      free(fifo_name);
    } else {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      p = isl_printer_print_ast_expr(p, local_index_packed);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
    }

    p = isl_printer_start_line(p);
    p = io_stmt_print_global_index(p, stmt);
    p = isl_printer_print_str(p, " = fifo_data;");
    p = isl_printer_end_line(p);
  }

  p = isl_printer_indent(p, -2);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "}");
  p = isl_printer_end_line(p);
  p = isl_printer_indent(p, 2);

  isl_ast_expr_free(local_index_packed);

  return p;
}

static __isl_give isl_printer *print_inter_trans_module_call(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_prog *prog, struct polysa_kernel *kernel,
  struct hls_info *hls, int arb, int boundary)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_print_str(p, "_inter_trans");
  if (boundary) 
    p = isl_printer_print_str(p, "_boundary");
  p = isl_printer_print_str(p, "(");
  p = print_module_arguments(p, prog, kernel, module, 0, hls->target, 1, arb, boundary);
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *print_intra_trans_module_call(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_prog *prog, struct polysa_kernel *kernel,
  struct hls_info *hls, int arb)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_print_str(p, "_intra_trans(");
  p = print_module_arguments(p, prog, kernel, module, 0, hls->target, 0, arb, 0);
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);

  return p;
}

/* Print the function call for intra_transfer module */
static __isl_give isl_printer *polysa_kernel_print_intra_trans(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.f.module;
  struct polysa_kernel *kernel = module->kernel;
  struct polysa_prog *prog = kernel->prog;

  if (module->double_buffer) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "if (arb == 0) {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);    
  }

  p = print_intra_trans_module_call(p, module, prog, kernel, hls, 0);

  if (module->double_buffer) {
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "} else {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
    
    p = print_intra_trans_module_call(p, module, prog, kernel, hls, 1);
    
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* Print the function call for inter_transfer module */
static __isl_give isl_printer *polysa_kernel_print_inter_trans(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.f.module;
  struct polysa_kernel *kernel = module->kernel;
  struct polysa_prog *prog = kernel->prog;
  int boundary = stmt->u.f.boundary;
   
  if (module->double_buffer) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "if (arb == 0) {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);    
  }

  p = print_inter_trans_module_call(p, module, prog, kernel, hls, 0, boundary);

  if (module->double_buffer) {
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "} else {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
    
    p = print_inter_trans_module_call(p, module, prog, kernel, hls, 1, boundary);
    
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* Print the function calls for inter_transfer and intra_tranfer modules */
static __isl_give isl_printer *polysa_kernel_print_inter_intra(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.f.module;
  struct polysa_kernel *kernel = module->kernel;
  struct polysa_prog *prog = kernel->prog;
  int boundary = stmt->u.f.boundary;

  if (module->double_buffer) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "if (arb == 0) {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
  }

  /* inter_trans */
  p = print_inter_trans_module_call(p, module, prog, kernel, hls, 0, boundary); 
  /* intra_trans */
  p = print_intra_trans_module_call(p, module, prog, kernel, hls, 0);
  
  if (module->double_buffer) {
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "} else {");
    p = isl_printer_end_line(p);

    p = isl_printer_indent(p, 4);
    
    /* inter_trans */
    p = print_inter_trans_module_call(p, module, prog, kernel, hls, 1, boundary);
    /* intra_trans */
    p = print_intra_trans_module_call(p, module, prog, kernel, hls, 1);

    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* Print the function calls for intra_transfer and inter_tranfer modules */
static __isl_give isl_printer *polysa_kernel_print_intra_inter(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.f.module;
  struct polysa_kernel *kernel = module->kernel;
  struct polysa_prog *prog = kernel->prog;
  int boundary = stmt->u.f.boundary;

  if (module->double_buffer) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "if (arb == 0) {");
    p = isl_printer_end_line(p);
    p = isl_printer_indent(p, 4);
  }

  /* intra_trans */
  p = print_intra_trans_module_call(p, module, prog, kernel, hls, 0);
  /* inter_trans */
  p = print_inter_trans_module_call(p, module, prog, kernel, hls, 0, boundary); 
  
  if (module->double_buffer) {
    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "} else {");
    p = isl_printer_end_line(p);

    p = isl_printer_indent(p, 4);
 
    /* intra_trans */
    p = print_intra_trans_module_call(p, module, prog, kernel, hls, 1);   
    /* inter_trans */
    p = print_inter_trans_module_call(p, module, prog, kernel, hls, 1, boundary);

    p = isl_printer_indent(p, -4);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "}");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* Print the state transfer for double buffers */
static __isl_give isl_printer *polysa_kernel_print_state_handle(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.f.module;
  isl_space *space;
  int n;

  if (module->in) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "intra_trans_en = 1;");
    p = isl_printer_end_line(p);
  } else {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "inter_trans_en = 1;");
    p = isl_printer_end_line(p);   
  }
    
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "arb = !arb;");
  p = isl_printer_end_line(p);

  if (module->in) {
    /* intra trans */
    space = module->intra_space;
  } else {
    /* inter trans */
    space = module->inter_space;
  }
  n = isl_space_dim(space, isl_dim_set);
  for (int i = 0; i < n; i++) {
    const char *name;
    name = isl_space_get_dim_name(space, isl_dim_set, i);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, "_prev = ");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);
  }

  return p;
}

static __isl_give isl_printer *print_delimiter(__isl_take isl_printer *p, 
  int *first)
{
  if (!(*first)) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \",\");");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  *first = 0;

  return p;
}

//static __isl_give isl_printer *print_fifo_name_prefix(__isl_take isl_printer * p, 
//  struct polysa_array_ref_group *group) 
//{
//  p = polysa_array_ref_group_print_fifo_name(group, p);
//
//  return p;
//}

/* Print out
 * "\/* [module_name] FIFO *\/"
 */
static __isl_give isl_printer *print_fifo_comment(__isl_take isl_printer *p, struct polysa_hw_module *module) 
{
  p = isl_printer_print_str(p, "/* ");
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_print_str(p, " fifo */");

  return p;
}

static __isl_give isl_printer *print_fifo_annotation(__isl_take isl_printer *p, 
  struct polysa_hw_module *module, struct polysa_array_ref_group *group, int in, int lower)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* fifo */ \");");
  p = isl_printer_end_line(p);

  return p;
}
  
/* Print out
 * "_[c]0_[c]1"
 */
static __isl_give isl_printer *print_inst_ids_suffix(__isl_take isl_printer *p,
  int n, __isl_keep isl_vec *offset)
{
  for (int i = 0; i < n; i++) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_\");");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_int(p, c");
    p = isl_printer_print_int(p, i);
    if (offset) {
      isl_val *val = isl_vec_get_element_val(offset, i);
      if (!isl_val_is_zero(val)) {
        p = isl_printer_print_str(p, " + ");
        p = isl_printer_print_val(p, val);
      }
      isl_val_free(val);
    }
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* Print out
 * "_[c0 + val]"
 * Increase the "pos"th index by the value of "val"
 */
static __isl_give isl_printer *print_inst_ids_inc_suffix(__isl_take isl_printer *p,
  int n, int pos, int val)
{
  for (int i = 0; i < n; i++) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_\");");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_int(p, c");
    p = isl_printer_print_int(p, i);
    if (i == pos) {
      if (val != 0) {
        p = isl_printer_print_str(p, " + ");
        p = isl_printer_print_int(p, val);
      }
    }
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);
  }

  return p;
}

/* "trans" maps the original inst ids to new inst ids.
 * This function first computes the preimage of inst ids under the trans map,
 * then prints out the preimage.
 * If the "offset" is set, it is added to the preimage.
 */
static __isl_give isl_printer *print_pretrans_inst_ids_suffix(__isl_take isl_printer *p,
  int n_id, __isl_keep isl_ast_expr *expr, __isl_keep isl_vec *offset)
{
  isl_ctx *ctx = isl_ast_expr_get_ctx(expr);
  int n;

  n = isl_ast_expr_op_get_n_arg(expr);

//  // debug
//  isl_printer *pd = isl_printer_to_file(ctx, stdout);
//  pd = isl_printer_set_output_format(pd, ISL_FORMAT_C);
//  printf("\n");
//  pd = isl_printer_print_ast_expr(pd, expr);
//  printf("\n");
//  // debug

  for (int i = 0; i < n_id; i++) {
    isl_ast_expr *expr_i = isl_ast_expr_get_op_arg(expr, i + 1);
    int format;
    
//    // debug
//    pd = isl_printer_print_ast_expr(pd, expr_i);
//    printf("\n");
//    // debug
  
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_\");");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_int(p, ");
    format = isl_printer_get_output_format(p);
    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
    p = isl_printer_print_ast_expr(p, expr_i);
    p = isl_printer_set_output_format(p, format);
    if (offset) {
      isl_val *val = isl_vec_get_element_val(offset, i);
      if (!isl_val_is_zero(val)) {
        p = isl_printer_print_str(p, " + ");
        p = isl_printer_print_val(p, val);
      }
      isl_val_free(val);
    }
    p = isl_printer_print_str(p, ");");
    p = isl_printer_end_line(p);

    isl_ast_expr_free(expr_i);
  }

  return p;
}

/* Print out
 * [fifo_name]_[module_name]
 */
static __isl_give isl_printer *print_fifo_prefix(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_array_ref_group *group)
{  
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"");
  p = polysa_array_ref_group_print_fifo_name(group, p);
  p = isl_printer_print_str(p, "_");
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_print_str(p, "\");");
  p = isl_printer_end_line(p);

  return p;
}

static char *build_io_module_lower_name(struct polysa_hw_module *module) 
{
  struct polysa_array_ref_group *group = module->io_groups[0];

  isl_printer *p = isl_printer_to_str(module->kernel->ctx);
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
  p = isl_printer_print_int(p, module->level - 1);
  if (module->in)
    p = isl_printer_print_str(p, "_in");
  else
    p = isl_printer_print_str(p, "_out");

  char *name = isl_printer_get_str(p);
  isl_printer_free(p);

  return name;
}

static __isl_give isl_printer *print_fifo_prefix_lower(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_array_ref_group *group)
{
  int lower_is_PE;

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"");
  p = polysa_array_ref_group_print_fifo_name(group, p);
  p = isl_printer_print_str(p, "_");
  assert(module->type != PE_MODULE);

  if (module->to_pe)
    lower_is_PE = 1;
  else
    lower_is_PE = 0;

  if (!lower_is_PE) {
    char *name = build_io_module_lower_name(module);
    p = isl_printer_print_str(p, name);
    free(name);
  } else {
    p = isl_printer_print_str(p, "PE");
  }
  p = isl_printer_print_str(p, "\");");
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_vec *get_trans_dir(__isl_keep isl_vec *dir, __isl_keep isl_mat *trans)
{
  isl_ctx *ctx = isl_mat_get_ctx(trans);
  int n_out, n_in;
  isl_vec *new_dir;

  n_out = isl_mat_rows(trans);
  n_in = isl_mat_cols(trans);
  new_dir = isl_vec_zero(ctx, n_out);

  for (int i = 0; i < n_out; i++) {
    isl_val *val = isl_vec_get_element_val(new_dir, i);
    for (int j = 0; j < n_in; j++) {
      isl_val *val2 = isl_vec_get_element_val(dir, j);
      isl_val *val3 = isl_mat_get_element_val(trans, i, j);
      val2 = isl_val_mul(val2, val3);
      val = isl_val_add(val, val2);
    }
    new_dir = isl_vec_set_element_val(new_dir, i, val);
  }

  return new_dir;
}

/* if module->type == PE_MODULE
 *   if boundary == 0:
 *     new_inst_id = io_trans(inst_id)
 *     print [fifo_name]_[module_name]_[new_inst_id]
 *   else if boundary == 1:
 *     new_inst_id = io_trans(inst_id)
 *     print [fifo_name]_[module_name]_[new_inst_id + dep_dir]
 * if module->type == IO_MODULE:
 *     print [fifo_name]_[module_name]_[inst_id]
 */
__isl_give isl_printer *print_fifo_decl(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct polysa_prog *prog, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.m.module;
  struct polysa_array_ref_group *group = stmt->u.m.group;
  int boundary = stmt->u.m.boundary;
  int n;
  int n_lane;

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Count channel number");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "fifo_cnt++;");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Print channel declarations of module: ");
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);      
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"");
  p = print_fifo_comment(p, module);
  p = isl_printer_print_str(p, " ");
  n_lane = get_io_group_n_lane(module, group);
//  if (module->type == PE_MODULE) {
//    n_lane = (group->local_array->array_type == POLYSA_EXT_ARRAY)? group->n_lane : 
//      ((group->group_type == POLYSA_DRAIN_GROUP)? group->n_lane: 
//      ((group->io_type == POLYSA_EXT_IO)? group->n_lane : group->io_buffers[0]->n_lane));
//  } else {
//    n_lane = module->data_pack_inter;
//  }
  if (hls->target == XILINX_HW)
    p = print_fifo_type_xilinx(p, group, n_lane);
  else if (hls->target == INTEL_HW)
    p = print_fifo_type_intel(p, group, n_lane);
  p = isl_printer_print_str(p, " ");
  p = polysa_array_ref_group_print_fifo_name(group, p);
  p = isl_printer_print_str(p, "_");
  p = isl_printer_print_str(p, module->name);
  p = isl_printer_print_str(p, "\");");
  p = isl_printer_end_line(p);

  n = isl_id_list_n_id(module->inst_ids);
  if (module->type == IO_MODULE || module->type == DRAIN_MODULE) {
    if (boundary) {
      p = print_inst_ids_inc_suffix(p, n, n - 1, 1);
    } else {
      p = print_inst_ids_suffix(p, n, NULL);
    }
  } else if (module->type == PE_MODULE) {
    if (boundary) 
      p = print_pretrans_inst_ids_suffix(p, n, group->io_L1_pe_expr, group->dir);
    else
      p = print_pretrans_inst_ids_suffix(p, n, group->io_L1_pe_expr, NULL); 
  }
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \";\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  if (hls->target == XILINX_HW) {
    /* Print fifo pragma */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
    p = isl_printer_end_line(p);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"#pragma HLS STREAM variable=");
    p = polysa_array_ref_group_print_fifo_name(group, p);
    p = isl_printer_print_str(p, "_");
    p = isl_printer_print_str(p, module->name);
    p = isl_printer_print_str(p, "\");");
    p = isl_printer_end_line(p);

    if (module->type == IO_MODULE || module->type == DRAIN_MODULE) {
      if (boundary) {
        p = print_inst_ids_inc_suffix(p, n, n - 1, 1);
      } else {
        p = print_inst_ids_suffix(p, n, NULL);
      }
    } else if (module->type == PE_MODULE) {
      if (boundary) {
        p = print_pretrans_inst_ids_suffix(p, n, group->io_L1_pe_expr, group->dir);
      } else {
        p = print_pretrans_inst_ids_suffix(p, n, group->io_L1_pe_expr, NULL);
      }
    }
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \" depth=2\");");
    p = isl_printer_end_line(p);
 
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
    p = isl_printer_end_line(p);   
  }

  return p;
}

__isl_give isl_printer *print_module_call_upper(__isl_take isl_printer *p, 
  struct polysa_kernel_stmt *stmt, struct polysa_prog *prog)
{
  struct polysa_hw_module *module = stmt->u.m.module;
  struct polysa_pe_dummy_module *pe_dummy_module = stmt->u.m.pe_dummy_module;
  int lower = stmt->u.m.lower;
  int upper = stmt->u.m.upper;
  int boundary = stmt->u.m.boundary;
  int dummy = stmt->u.m.dummy;
  int first = 1;  
  int n;
  char *module_name = stmt->u.m.module_name;
  isl_space *space;

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "// Print calls of module: ");
  p = isl_printer_print_str(p, module_name);
  if (boundary) {
    p = isl_printer_print_str(p, "_boundary");
  }
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"");
  p = isl_printer_print_str(p, module_name);
  if (boundary) {
    p = isl_printer_print_str(p, "_boundary");
  }
  p = isl_printer_print_str(p, "(\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_indent(p, 4);");
  p = isl_printer_end_line(p);

  /* module identifiers */
  if (!dummy) {
    for (int i = 0; i < isl_id_list_n_id(module->inst_ids); i++) {
      p = print_delimiter(p, &first); 
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* module id */ \");");
      p = isl_printer_end_line(p);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_int(p, c");
      p = isl_printer_print_int(p, i);
      p = isl_printer_print_str(p, ");");
      p = isl_printer_end_line(p);
    }
  } else {
    isl_ast_expr *expr = pe_dummy_module->io_group->io_L1_pe_expr;
    int n_arg = isl_ast_expr_op_get_n_arg(expr);

    for (int i = 0; i < isl_id_list_n_id(module->inst_ids); i++) {
      int format;
      p = print_delimiter(p, &first);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* module id */ \");");
      p = isl_printer_end_line(p);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_int(p, ");

      isl_ast_expr *expr_i = isl_ast_expr_get_op_arg(expr, i + 1);
      p = isl_printer_print_ast_expr(p, expr_i);
      isl_ast_expr_free(expr_i);
    
      p = isl_printer_print_str(p, ");");
      p = isl_printer_end_line(p);
    }
  }

  /* params */
  space = isl_union_set_get_space(module->kernel->arrays);
  n = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < n; i++) {
    p = print_delimiter(p, &first);

    const char *name = isl_space_get_dim_name(space, isl_dim_set, i);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* param */");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, "\");");
    p = isl_printer_end_line(p);
  }
  isl_space_free(space);

  /* host iterators */
  n = isl_space_dim(module->kernel->space, isl_dim_set);
  for (int i = 0; i < n; i++) {
    p = print_delimiter(p, &first);

    const char *name = isl_space_get_dim_name(module->kernel->space, isl_dim_set, i);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* host iter */ ");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, "\");");
    p = isl_printer_end_line(p);
  }

  /* scalar and arrays */
  if (module->type != PE_MODULE && module->level == 3) {
    p = print_delimiter(p, &first);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* array */ ");
    p = isl_printer_print_str(p, module->io_groups[0]->array->name);
    p = isl_printer_print_str(p, "\");");
    p = isl_printer_end_line(p);
  } else if (module->type == PE_MODULE) {
    for (int i = 0; i < prog->n_array; i++) {
      int required;

      required = polysa_kernel_requires_array_argument(module->kernel, i);
      if (required < 0)
        return isl_printer_free(p);
      if (!required)
        continue;

      if (polysa_array_is_read_only_scalar(&prog->array[i])) {
        p = print_delimiter(p, &first);
    
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* scalar */ ");
        p = isl_printer_print_str(p, module->io_groups[0]->array->name);
        p = isl_printer_print_str(p, "\");");
        p = isl_printer_end_line(p);
      }
    }
  }

  /* FIFO */
  n = isl_id_list_n_id(module->inst_ids);
  if (module->type == PE_MODULE) {
    if (dummy) {
      struct polysa_array_ref_group *group = pe_dummy_module->io_group;
      p = print_delimiter(p, &first);
      p = print_fifo_annotation(p, module, group, 1, 0);
      p = print_fifo_prefix(p, module, group);
      if (isl_vec_is_zero(group->dir)) {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_in\")");
        p = isl_printer_end_line(p);
      }
//      p = print_inst_ids_suffix(p, n, group->dir);
      p = print_pretrans_inst_ids_suffix(p, n, group->io_L1_pe_expr, group->dir);
    } else {
      for (int i = 0; i < module->n_io_group; i++) {
        struct polysa_array_ref_group *group = module->io_groups[i];
        if (group->pe_io_dir == IO_INOUT) {
          p = print_delimiter(p, &first);
          p = print_fifo_annotation(p, module, group, 1, 0);
          p = print_fifo_prefix(p, module, group);
          if (isl_vec_is_zero(group->dir)) {
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_in\")");
            p = isl_printer_end_line(p);
          }
          p = print_inst_ids_suffix(p, n, NULL);
  
          p = print_delimiter(p, &first);
          p = print_fifo_annotation(p, module, group, 0, 0);
          p = print_fifo_prefix(p, module, group);
          if (isl_vec_is_zero(group->dir)) {
            p = isl_printer_start_line(p);
            p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"_out\")");
            p = isl_printer_end_line(p);
          }
          p = print_inst_ids_suffix(p, n, group->dir);
        } else {
          p = print_delimiter(p, &first);
          p = print_fifo_annotation(p, module, group, group->pe_io_dir == IO_IN? 1 : 0, 0);
          p = print_fifo_prefix(p, module, group);
          p = print_inst_ids_suffix(p, n, NULL);
        }
      }
    }
  } else {
    if (!module->to_mem) {
      for (int i = 0; i < module->n_io_group; i++) {
        struct polysa_array_ref_group *group = module->io_groups[i]; 
//        if (!(boundary && !module->in)) {
//          p = print_delimiter(p, &first);
//          p = print_fifo_annotation(p, module, group, 1, 0);
//          p = print_fifo_prefix(p, module, group);
//          p = print_inst_ids_suffix(p, n, NULL);
//        }
//
//        if (!(boundary && module->in)) {
//          p = print_delimiter(p, &first);
//          p = print_fifo_annotation(p, module, group, 0, 0);
//          p = print_fifo_prefix(p, module, group);
//          p = print_inst_ids_inc_suffix(p, n, n - 1, 1);
//        }
        if (module->in) {
          p = print_delimiter(p, &first);
          p = print_fifo_annotation(p, module, group, 1, 0);
          p = print_fifo_prefix(p, module, group);
          p = print_inst_ids_suffix(p, n, NULL);
  
          if (!boundary) {
            p = print_delimiter(p, &first);
            p = print_fifo_annotation(p, module, group, 0, 0);
            p = print_fifo_prefix(p, module, group);
            p = print_inst_ids_inc_suffix(p, n, n - 1, 1);
          }
        } else {
          if (!boundary) {
            p = print_delimiter(p, &first);
            p = print_fifo_annotation(p, module, group, 0, 0);
            p = print_fifo_prefix(p, module, group);
            p = print_inst_ids_inc_suffix(p, n, n - 1, 1);
          }

          p = print_delimiter(p, &first);
          p = print_fifo_annotation(p, module, group, 1, 0);
          p = print_fifo_prefix(p, module, group);
          p = print_inst_ids_suffix(p, n, NULL);
        }
      }
    }
  }
  
  return p;
}

static __isl_give isl_printer *print_module_call_lower(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct polysa_prog *prog)
{
  struct polysa_hw_module *module = stmt->u.m.module;
  int lower = stmt->u.m.lower;
  int first = 0;
  int n = isl_id_list_n_id(module->inst_ids);
  int lower_is_PE;
  int boundary = stmt->u.m.boundary;

  if (lower) {
    struct polysa_array_ref_group *group = module->io_groups[0];

    p = print_delimiter(p, &first);

    p = print_fifo_annotation(p, module, group, module->in? 0 : 1, 1); 
    p = print_fifo_prefix_lower(p, module, group);
    if (module->to_pe)
      lower_is_PE = 1;
    else
      lower_is_PE = 0;

    if (lower_is_PE) {
//      if (module->in)
//        p = print_pretrans_inst_ids_suffix(p, module->kernel->n_sa_dim, boundary? group->io_pe_expr_boundary : group->io_pe_expr, NULL);
//      else {
//        if (module->level == 1)
//          p = print_pretrans_inst_ids_suffix(p, module->kernel->n_sa_dim, boundary? group->io_pe_expr_boundary : group->io_pe_expr, NULL);
//        else
//          p = print_pretrans_inst_ids_suffix(p, module->kernel->n_sa_dim, boundary? group->io_pe_expr_boundary : group->io_pe_expr, group->dir);
//      }
      p = print_pretrans_inst_ids_suffix(p, module->kernel->n_sa_dim, boundary? group->io_pe_expr_boundary : group->io_pe_expr, NULL);
    } else {
//      if (module->in)
//        p = print_inst_ids_suffix(p, n + 1, NULL);
//      else {
//        p = print_inst_ids_inc_suffix(p, n + 1, n, 1);
//      }
      p = print_inst_ids_suffix(p, n + 1, NULL);
    }
  } 

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_indent(p, -4);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \");\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  return p;
}

__isl_give isl_printer *polysa_kernel_print_module_call(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct polysa_prog *prog)
{
  int upper = stmt->u.m.upper;
  int lower = stmt->u.m.lower;
  int complete = (upper == 0 && lower == 0);
  int dummy = stmt->u.m.dummy;
  int boundary = stmt->u.m.boundary;
  char *module_name = stmt->u.m.module_name;
  struct polysa_hw_module *module = stmt->u.m.module;
  p = ppcg_start_block(p);

  /* Build the module name */  
  if (complete) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "// Count module number");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, module_name);
    if (boundary) 
      p = isl_printer_print_str(p, "_boundary");
    p = isl_printer_print_str(p, "_cnt++;");
    p = isl_printer_end_line(p);
    if (module->is_filter && module->is_buffer) {
      /* Print counter for inter_trans and intra_trans module */
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, module_name);
      p = isl_printer_print_str(p, "_intra_trans_cnt++;");
      p = isl_printer_end_line(p);

      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, module_name);
      if (boundary)        
        p = isl_printer_print_str(p, "_inter_trans_boundary_cnt++;");
      else
        p = isl_printer_print_str(p, "_inter_trans_cnt++;");
      p = isl_printer_end_line(p);
    }

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
    p = isl_printer_end_line(p);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* Module Call */\");");
    p = isl_printer_end_line(p);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
    p = isl_printer_end_line(p);

    p = print_module_call_upper(p, stmt, prog);
    p = print_module_call_lower(p, stmt, prog);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
    p = isl_printer_end_line(p);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* Module Call */\");");
    p = isl_printer_end_line(p);
    
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
    p = isl_printer_end_line(p);
  } else {
    if (upper) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "// Count module number");
      p = isl_printer_end_line(p);
  
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, module_name);
      if (boundary) 
        p = isl_printer_print_str(p, "_boundary");
      p = isl_printer_print_str(p, "_cnt++;");
      p = isl_printer_end_line(p);
      if (module->is_filter && module->is_buffer) {
        /* Print counter for inter_trans and intra_trans module */
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, module_name);
        p = isl_printer_print_str(p, "_intra_trans_cnt++;");
        p = isl_printer_end_line(p);

        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, module_name);
        if (boundary)
          p = isl_printer_print_str(p, "_inter_trans_boundary_cnt++;");
        else
          p = isl_printer_print_str(p, "_inter_trans_cnt++;");
        p = isl_printer_end_line(p);
      }

      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
      p = isl_printer_end_line(p);
      
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* Module Call */\");");
      p = isl_printer_end_line(p);
      
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
      p = isl_printer_end_line(p);

      p = print_module_call_upper(p, stmt, prog);
    } else { 
      p = print_module_call_lower(p, stmt, prog);

      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
      p = isl_printer_end_line(p);
      
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* Module Call */\");");
      p = isl_printer_end_line(p);
      
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
      p = isl_printer_end_line(p);

      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
      p = isl_printer_end_line(p);
    }
  }

  p = ppcg_end_block(p);

  return p;
}

static __isl_give isl_printer *polysa_kernel_print_fifo_decl(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct polysa_prog *prog, struct hls_info *hls)
{
  p = ppcg_start_block(p);

  /* Build the fifo_decl */
  p = print_fifo_decl(p, stmt, prog, hls);

  p = ppcg_end_block(p);

  return p;
}

/* Print an I/O statement.
 *
 * An in I/O statement is printed as
 *
 *  local[] = fifo.read(); 
 *
 * while an out I/O statement is printed as
 *
 *  fifo.write(local);
 */
static __isl_give isl_printer *polysa_kernel_print_io(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.i.module;
  struct polysa_array_ref_group *group = stmt->u.i.group;
  char *fifo_name;
  isl_ctx *ctx = isl_printer_get_ctx(p);
  int is_dummy = stmt->u.i.dummy;
  fifo_name = concat(ctx, stmt->u.i.fifo_name, stmt->u.i.in == 1? "in" : "out");
  int data_pack = stmt->u.i.data_pack;
  
  if (is_dummy) {
    /* [type] fifo_data; */
    p = isl_printer_start_line(p);
    if (data_pack == 1) {
      p = isl_printer_print_str(p, group->array->type);
    } else {
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "_t");
      p = isl_printer_print_int(p, data_pack);
    }
    p = isl_printer_print_str(p, " fifo_data;");
    p = isl_printer_end_line(p);

    /* fifo_data = fifo.read() */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_data = ");
    if (hls->target == XILINX_HW) 
      p = print_fifo_rw_xilinx(p, fifo_name, 1);
    else if (hls->target == INTEL_HW) 
      p = print_fifo_rw_intel(p, fifo_name, 1);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);

    free(fifo_name);
    return p;
  }

  int nxt_data_pack = stmt->u.i.nxt_data_pack;
  isl_ast_expr *local_index_packed;
  isl_ast_expr *arg, *div;
  int n_arg;
  local_index_packed = isl_ast_expr_copy(stmt->u.i.local_index);
  /* Modify the local index */
  if (data_pack > 1) {
    n_arg = isl_ast_expr_get_op_n_arg(local_index_packed);
    arg = isl_ast_expr_get_op_arg(local_index_packed, n_arg - 1);
    div = isl_ast_expr_from_val(isl_val_int_from_si(ctx, data_pack));
    arg = isl_ast_expr_div(arg, div);
    local_index_packed = isl_ast_expr_set_op_arg(local_index_packed, n_arg - 1, arg);
  }

  if (data_pack == nxt_data_pack) {
    p = isl_printer_start_line(p);
    if (stmt->u.i.in) {
      p = isl_printer_print_ast_expr(p, local_index_packed); 
      p = isl_printer_print_str(p, " = ");
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 1);
      else if (hls->target == INTEL_HW)
        p = print_fifo_rw_intel(p, fifo_name, 1);
    } else {
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 0);
      else if (hls->target == INTEL_HW)
        p = print_fifo_rw_intel(p, fifo_name, 0);
      p = isl_printer_print_ast_expr(p, local_index_packed);
      p = isl_printer_print_str(p, ")");
    }
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);
  } else {
    p = ppcg_start_block(p);

    /* [type] fifo_data; */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, group->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, data_pack);
    p = isl_printer_print_str(p, " fifo_data;");
    p = isl_printer_end_line(p);

    if (stmt->u.i.in) {
      /* fifo_data = fifo.read(); */
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 1);
      else if (hls->target == INTEL_HW)
        p = print_fifo_rw_intel(p, fifo_name, 1);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);      

      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "for (int n = 0; n < ");
      p = isl_printer_print_int(p, data_pack / nxt_data_pack);
      p = isl_printer_print_str(p, "; n++) {");
      p = isl_printer_end_line(p);
      if (hls->target == XILINX_HW) {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "#pragma HLS UNROLL");
        p = isl_printer_end_line(p);
      }
      
      p = isl_printer_indent(p, 4);
      if (hls->target == XILINX_HW) {
        isl_ast_expr *op;
        isl_ast_expr *expr = stmt->u.i.local_index; 
        int n_arg = isl_ast_expr_op_get_n_arg(expr);
//        // debug
//        isl_printer *pd = isl_printer_to_file(isl_printer_get_ctx(p), stdout);
//        pd = isl_printer_print_ast_expr(pd, expr);
//        printf("\n");
//        // debug

        p = isl_printer_start_line(p);
        op = isl_ast_expr_op_get_arg(expr, 0);
        p = isl_printer_print_ast_expr(p, op); // array_name
        isl_ast_expr_free(op);
        for (int i = 0; i < n_arg - 1; i++) {
          op = isl_ast_expr_op_get_arg(expr, 1 + i);
          p = isl_printer_print_str(p, "[");
          if (i == n_arg - 2) {
            p = isl_printer_print_str(p, "n");
          } else {
            p = isl_printer_print_ast_expr(p, op);
          }
          p = isl_printer_print_str(p, "]");
          isl_ast_expr_free(op);
        }

        p = isl_printer_print_str(p, " = ");
        if (nxt_data_pack == 1) {
          p = isl_printer_print_str(p, "Reinterpret<");
          p = isl_printer_print_str(p, group->array->type);
          p = isl_printer_print_str(p, ">(");
          p = isl_printer_print_str(p, "(ap_uint<");
          p = isl_printer_print_int(p, group->array->size * 8 * nxt_data_pack);
          p = isl_printer_print_str(p, ">)");
        }
        p = isl_printer_print_str(p, "fifo_data(");
        p = isl_printer_print_int(p, group->array->size * 8 * nxt_data_pack - 1);
        p = isl_printer_print_str(p, ", 0)");
        if (nxt_data_pack == 1) {
          p = isl_printer_print_str(p, ")");
        }
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
  
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_data = fifo_data >> ");
        p = isl_printer_print_int(p, group->array->size * 8 * nxt_data_pack);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      } else if (hls->target == INTEL_HW) {
        // TODO
      }
  
      p = isl_printer_indent(p, -4);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "}");
      p = isl_printer_end_line(p);
    } else {
      if (hls->target == XILINX_HW) {
        int first = 1;
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "fifo_data = (");
        for (int i = data_pack / nxt_data_pack - 1; i >= 0; i--) {
          isl_ast_expr *expr = stmt->u.i.local_index;
          isl_ast_expr *op;
          int n_arg = isl_ast_expr_op_get_n_arg(expr);

          if (!first) 
            p = isl_printer_print_str(p, ", ");
          // print localA[][ * data_pack + i]
          
          if (nxt_data_pack == 1) {
            p = isl_printer_print_str(p, "Reinterpret<ap_uint<");
            p = isl_printer_print_int(p, group->array->size * 8 * nxt_data_pack);
            p = isl_printer_print_str(p, "> >(");
          }          

          op = isl_ast_expr_op_get_arg(expr, 0);
          p = isl_printer_print_ast_expr(p, op);
          isl_ast_expr_free(op);
          for (int j = 0; j < n_arg - 1; j++) {
            op = isl_ast_expr_op_get_arg(expr, 1 + j);
            p = isl_printer_print_str(p, "[");
            if (j == n_arg - 2) {
              p = isl_printer_print_int(p, i);
            } else {
              p = isl_printer_print_ast_expr(p, op);
            }
            p = isl_printer_print_str(p, "]");
            isl_ast_expr_free(op);
          }

          if (nxt_data_pack == 1) {
            p = isl_printer_print_str(p, ")");
          }

          first = 0;
        }
        p = isl_printer_print_str(p, ");");
        p = isl_printer_end_line(p);
        
        p = isl_printer_start_line(p);
        p = print_fifo_rw_xilinx(p, fifo_name, 0);
        p = isl_printer_print_str(p, "fifo_data);");
        p = isl_printer_end_line(p);
      } else if (hls->target == INTEL_HW) {
        // TODO
      }
    }

    p = ppcg_end_block(p);
  }
  free(fifo_name); 
  isl_ast_expr_free(local_index_packed);
  return p;
}

/* Print an I/O statement.
 *
 * An in I/O statement is printed as
 *
 *  [type] fifo_data;
 *  fifo_data = fifo.read();
 *  if (filter_condition) {
 *    local = fifo_data; // if buf == 1
 *    fifo_local.write(fifo_data); // if buf == 0
 *  } else {
 *    fifo.write(fifo_data);
 *  }
 *
 * if filter_depth < 0
 *
 *  [type] fifo_data;
 *  fifo_data = fifo.read();
 *  local = fifo_data; // if buf == 1
 *  fifo_local.write(fifo_data); // if buf == 0
 *
 * An out I/O staement is printed as 
 *
 *  if (filter_condition) {
 *    fifo_data = local;
 *  } else {
 *    fifo_data = fifo.read();
 *  }
 *  fifo.write(fifo_data);
 */
static __isl_give isl_printer *polysa_kernel_print_io_transfer_default(
  __isl_take isl_printer *p, struct polysa_kernel_stmt *stmt,
  struct polysa_array_ref_group *group, int n_lane, struct hls_info *hls)
{
  isl_ctx *ctx;
  char *fifo_name;
  ctx = isl_printer_get_ctx(p);
  int boundary = stmt->u.i.boundary;
  /* If the statement is a boundary statement, 
   * then ignore the filter condition by setting filter_sched_depth as -1
   */
  if (boundary) 
    stmt->u.i.filter_sched_depth = -1;

  isl_ast_expr *local_index_packed;
  isl_ast_expr *arg, *div;
  local_index_packed = isl_ast_expr_copy(stmt->u.i.local_index);
  int n_arg;
  /* Modify the local index */
  if (n_lane > 1) {
    n_arg = isl_ast_expr_get_op_n_arg(local_index_packed);
    arg = isl_ast_expr_get_op_arg(local_index_packed, n_arg - 1);
    div = isl_ast_expr_from_val(isl_val_int_from_si(ctx, n_lane));
    arg = isl_ast_expr_div(arg, div);
    local_index_packed = isl_ast_expr_set_op_arg(local_index_packed, n_arg - 1, arg);
  }

  /* Declare the fifo data variable */
  p = isl_printer_start_line(p);
  if (n_lane == 1) {
    p = isl_printer_print_str(p, stmt->u.i.array->type);
  } else {
    p = isl_printer_print_str(p, stmt->u.i.array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, n_lane);
  }
  p = isl_printer_print_str(p, " fifo_data;");
  p = isl_printer_end_line(p);

  if (stmt->u.i.in) {
    fifo_name = concat(ctx, stmt->u.i.fifo_name, "in");
    /* fifo_data = fifo.read() */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_data");
    p = isl_printer_print_str(p, " = ");
    if (hls->target == XILINX_HW)
      p = print_fifo_rw_xilinx(p, fifo_name, 1);
    else 
      p = print_fifo_rw_intel(p, fifo_name, 1);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);
    free(fifo_name);

    /* if (filter_condition) */
    if (stmt->u.i.filter_sched_depth >= 0) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "if (c");
      p = isl_printer_print_int(p, stmt->u.i.filter_sched_depth);
      p = isl_printer_print_str(p, " == p"); 
      p = isl_printer_print_int(p, stmt->u.i.filter_param_id);
      p = isl_printer_print_str(p, ") {");
      p = isl_printer_end_line(p);
      p = isl_printer_indent(p, 2);
    }
      
    if (stmt->u.i.buf) {
      /* local[][] = fifo_data */
      p = isl_printer_start_line(p);
//      p = io_stmt_print_local_index(p, stmt);
      p = isl_printer_print_ast_expr(p, local_index_packed);
      p = isl_printer_print_str(p, " = fifo_data;");
      p = isl_printer_end_line(p);
    } else {
      /* fifo_local.write(fifo_data); */
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "local_out");
      p = isl_printer_start_line(p);
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 0);
      else 
        p = print_fifo_rw_intel(p, fifo_name, 0);
      p = isl_printer_print_str(p, "fifo_data);");
      p = isl_printer_end_line(p);
      free(fifo_name);
    }
    
    if (stmt->u.i.filter_sched_depth >= 0) {
      p = isl_printer_indent(p, -2);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "} else {");
      p = isl_printer_end_line(p);
      p = isl_printer_indent(p, 2);
    
      /* fifo.write(fifo_data); */
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "out");
      p = isl_printer_start_line(p);
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 0);
      else
        p = print_fifo_rw_intel(p, fifo_name, 0);
      p = isl_printer_print_str(p, "fifo_data);");
      p = isl_printer_end_line(p);
      free(fifo_name);

      p = isl_printer_indent(p, -2);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "}");
      p = isl_printer_end_line(p);
    }
  } else {
    /* if (filter_condition) */
    if (stmt->u.i.filter_sched_depth >= 0) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "if (c");
      p = isl_printer_print_int(p, stmt->u.i.filter_sched_depth);
      p = isl_printer_print_str(p, " == p"); 
      p = isl_printer_print_int(p, stmt->u.i.filter_param_id);
      p = isl_printer_print_str(p, ") {");
      p = isl_printer_end_line(p);
      p = isl_printer_indent(p, 2);
    }

    if (stmt->u.i.buf) {
      /* fifo_data = local[][] */
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      p = isl_printer_print_ast_expr(p, local_index_packed);
//      p = io_stmt_print_local_index(p, stmt);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
    } else {
      /* fifo_data = fifo_local.read() */
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "local_in");
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 1);
      else 
        p = print_fifo_rw_intel(p, fifo_name, 1);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
      free(fifo_name);
    }

    if (stmt->u.i.filter_sched_depth >= 0) {
      p = isl_printer_indent(p, -2);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "} else {");
      p = isl_printer_end_line(p);
      p = isl_printer_indent(p, 2);
    
      /* fifo_data = fifo.read() */
      fifo_name = concat(ctx, stmt->u.i.fifo_name, "in");
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "fifo_data = ");
      if (hls->target == XILINX_HW)
        p = print_fifo_rw_xilinx(p, fifo_name, 1);
      else 
        p = print_fifo_rw_intel(p, fifo_name, 1);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
      free(fifo_name);
    
      p = isl_printer_indent(p, -2);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "}");
      p = isl_printer_end_line(p);
    }

    /* fifo.write(fifo_data) */
    fifo_name = concat(ctx, stmt->u.i.fifo_name, "out");
    p = isl_printer_start_line(p);
    if (hls->target == XILINX_HW)
      p = print_fifo_rw_xilinx(p, fifo_name, 0);
    else 
      p = print_fifo_rw_intel(p, fifo_name, 0);
    p = isl_printer_print_str(p, "fifo_data);");
    p = isl_printer_end_line(p);
    free(fifo_name);
  }

  isl_ast_expr_free(local_index_packed);

  return p;
}

/* Print an I/O statement.
 * is_filter = 0
 * is_buf = 1
 * An in I/O statement is printed as
 *
 *  [type] fifo_data;
 *  [type2] buf_data;
 *  [type] buf_data_split[];
 *  buf_data = local_buf[...];
 *  fifo_data = fifo.read();
 *  for (int n = 0; n < n_lane / nxt_n_lane; n++) {
 *    buf_data_split[n] = buf_data();
 *    buf_data = buf_data >> DW;
 *  }
 *  buf_data_split[...] = Reinterpret<>(fifo_data);
 *  buf_data = (buf_data_split[1], ...);
 *  local_buf[...] = buf_data;
 *
 * An out I/O staement is printed as 
 *
 *  [type] fifo_data;
 *  [type2] buf_data;
 *  [type] buf_data_split[];
 *  buf_data = local_buf[...];
 *  for (int n = 0; n < n_lane / nxt_n_lane; n++) {
 *    buf_data_split[n] = buf_data();
 *    buf_data = buf_data >> DW;
 *  }
 *  fifo_data = Reinterpret<>(buf_data_split[...]);
 *  fifo.write(fifo_data);
 */
static __isl_give isl_printer *polysa_kernel_print_io_transfer_data_pack(
  __isl_take isl_printer *p, struct polysa_kernel_stmt *stmt,
  struct polysa_array_ref_group *group, int n_lane, int nxt_n_lane, struct hls_info *hls)
{
  isl_ctx *ctx;
  ctx = isl_printer_get_ctx(p);
  int boundary = stmt->u.i.boundary;

  char *fifo_name;
  isl_ast_expr *expr, *op;
  int n_arg;
  int r;
  isl_val *val;
  isl_ast_expr *local_index_packed;
  isl_ast_expr *arg, *div;
  local_index_packed = isl_ast_expr_copy(stmt->u.i.local_index);
  /* Modify the local index */
  if (n_lane > 1) {
    n_arg = isl_ast_expr_get_op_n_arg(local_index_packed);
    arg = isl_ast_expr_get_op_arg(local_index_packed, n_arg - 1);
    div = isl_ast_expr_from_val(isl_val_int_from_si(ctx, n_lane));
    arg = isl_ast_expr_div(arg, div);
    local_index_packed = isl_ast_expr_set_op_arg(local_index_packed, n_arg - 1, arg);
  }

  /* [type] fifo_data; */
  p = isl_printer_start_line(p);
  if (nxt_n_lane == 1) {
    p = isl_printer_print_str(p, group->array->type);
  } else {
    p = isl_printer_print_str(p, group->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, nxt_n_lane);
  }
  p = isl_printer_print_str(p, " ");
  p = isl_printer_print_str(p, "fifo_data;");
  p = isl_printer_end_line(p);

  /* [type2] buf_data */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, group->array->name);
  p = isl_printer_print_str(p, "_t");
  p = isl_printer_print_int(p, n_lane);
  p = isl_printer_print_str(p, " ");
  p = isl_printer_print_str(p, "buf_data;");
  p = isl_printer_end_line(p);

  /* [type] buf_data_split[] */
  p = isl_printer_start_line(p);
  if (nxt_n_lane == 1) {
    p = isl_printer_print_str(p, "ap_uint<");
    p = isl_printer_print_int(p, group->array->size * 8);
    p = isl_printer_print_str(p, ">");
  } else {
    p = isl_printer_print_str(p, group->array->name);
    p = isl_printer_print_str(p, "_t");
    p = isl_printer_print_int(p, nxt_n_lane);
  }
  p = isl_printer_print_str(p, " buf_data_split[");
  p = isl_printer_print_int(p, n_lane / nxt_n_lane);
  p = isl_printer_print_str(p, "];");
  p = isl_printer_end_line(p);
  if (hls->target == XILINX_HW) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS ARRAY_PARTITION variable=buf_data_split complete");
    p = isl_printer_end_line(p);
  }
    
  /* buf_data = local[] */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "buf_data = ");
  p = isl_printer_print_ast_expr(p, local_index_packed);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "for (int n = 0; n < ");
  p = isl_printer_print_int(p, n_lane / nxt_n_lane);
  p = isl_printer_print_str(p, "; n++) {");
  p = isl_printer_end_line(p);

  p = isl_printer_indent(p, 4);
  if (hls->target == XILINX_HW) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "#pragma HLS UNROLL");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "buf_data_split[n] = buf_data(");
    p = isl_printer_print_int(p, group->array->size * 8 * nxt_n_lane - 1);
    p = isl_printer_print_str(p, ", 0);");
    p = isl_printer_end_line(p);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "buf_data = buf_data >> ");
    p = isl_printer_print_int(p, group->array->size * 8 * nxt_n_lane);
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);
  } else if (hls->target = INTEL_HW) {
    // TODO
  }

  p = isl_printer_indent(p, -4);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "}");
  p = isl_printer_end_line(p);

  /* split_i = ... */
  expr = isl_ast_expr_copy(stmt->u.i.local_index);
  n_arg = isl_ast_expr_op_get_n_arg(expr);
  op = isl_ast_expr_op_get_arg(expr, n_arg - 1);
  r = n_lane / nxt_n_lane;
  val = isl_val_int_from_si(ctx, nxt_n_lane);
  op = isl_ast_expr_div(op, isl_ast_expr_from_val(val));
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "int split_i = (");
  p = isl_printer_print_ast_expr(p, op);
  p = isl_printer_print_str(p, ") % ");
  p = isl_printer_print_int(p, r);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_ast_expr_free(op);
  isl_ast_expr_free(expr);

  if (stmt->u.i.in) {
    fifo_name = concat(ctx, stmt->u.i.fifo_name, "in");

    /* fifo_data = fifo.read */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_data = ");
    if (hls->target == XILINX_HW) {
      p = print_fifo_rw_xilinx(p, fifo_name, 1);
    } else if (hls->target == INTEL_HW) {
      p = print_fifo_rw_intel(p, fifo_name, 1);
    }
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);

    /* buf_data_split[...] = Reinterpret<>(fifo_data); */   
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "buf_data_split[split_i] = ");
    if (hls->target == XILINX_HW) {
      if (nxt_n_lane == 1) {
        p = isl_printer_print_str(p, "Reinterpret<ap_uint<");
        p = isl_printer_print_int(p, group->array->size * 8);
        p = isl_printer_print_str(p, "> >(fifo_data);");
      } else {
        p = isl_printer_print_str(p, "fifo_data;");
      }
    } else if (hls->target == INTEL_HW) {
      // TODO
      printf("[PolySA] Codegen for Intel devices is not supported yet!\n");
    }    
    p = isl_printer_end_line(p);

    /* buf_data = (buf_data_split[1], ...); */
    p = isl_printer_start_line(p);
    if (hls->target == XILINX_HW) {
      int first = 1;
      p = isl_printer_print_str(p, "buf_data = (");
      for (int i = n_lane / nxt_n_lane - 1; i >= 0; i--) {
        if (!first)
          p = isl_printer_print_str(p, ", ");
        p = isl_printer_print_str(p, "buf_data_split[");
        p = isl_printer_print_int(p, i);
        p = isl_printer_print_str(p, "]");

        first = 0;
      }
      p = isl_printer_print_str(p, ");");
    } else if (hls->target = INTEL_HW) {
      // TODO
      printf("[PolySA] Codegen for Intel devices is not supported yet!\n");
    }
    p = isl_printer_end_line(p);

    /* local_buf[...] = buf_data; */
    p = isl_printer_start_line(p);
    p = isl_printer_print_ast_expr(p, local_index_packed);
    p = isl_printer_print_str(p, " = buf_data;");
    p = isl_printer_end_line(p);

    free(fifo_name);
  } else {
    fifo_name = concat(ctx, stmt->u.i.fifo_name, "out");

    /* fifo_data = Reinterpret<>(buf_data_split[...]); */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_data = ");
    if (hls->target == XILINX_HW) {
      if (nxt_n_lane == 1) {
        p = isl_printer_print_str(p, "Reinterpret<");
        p = isl_printer_print_str(p, group->array->type);
        p = isl_printer_print_str(p, ">(");
      } 
      p = isl_printer_print_str(p, "buf_data_split[split_i]");
      if (nxt_n_lane == 1) {
        p = isl_printer_print_str(p, ")");
      }
    } else if (hls->target == INTEL_HW) {
      // TODO
      printf("[PolySA] Codegen for Intel devices is not supported yet!\n");
    }
    p = isl_printer_print_str(p, ";");
    p = isl_printer_end_line(p);

    /* fifo.write(fifo_data); */
    p = isl_printer_start_line(p);
    if (hls->target == XILINX_HW) {
      p = print_fifo_rw_xilinx(p, fifo_name, 0);
    } else if (hls->target == INTEL_HW) {
      p = print_fifo_rw_intel(p, fifo_name, 0);
    }
    p = isl_printer_print_str(p, "fifo_data);");
    p = isl_printer_end_line(p);

    free(fifo_name);
  }

  isl_ast_expr_free(local_index_packed);

  return p;
}

/* Print an I/O statement.
 */
__isl_give isl_printer *polysa_kernel_print_io_transfer(__isl_take isl_printer *p,
  struct polysa_kernel_stmt *stmt, struct hls_info *hls)
{
  struct polysa_hw_module *module = stmt->u.i.module;
  struct polysa_array_ref_group *group = stmt->u.i.group;
  int n_lane = stmt->u.i.data_pack;
  int nxt_n_lane = stmt->u.i.nxt_data_pack;
  int is_filter = stmt->u.i.filter;
  int is_buf = stmt->u.i.buf;
  isl_ctx *ctx = isl_printer_get_ctx(p);

  p = ppcg_start_block(p); 
  if (n_lane == nxt_n_lane) {
    p = polysa_kernel_print_io_transfer_default(p, stmt, group, n_lane, hls);
  } else {
    p = polysa_kernel_print_io_transfer_data_pack(p, stmt, group, n_lane, nxt_n_lane, hls);
  }
  p = ppcg_end_block(p);

  return p;
}

static __isl_give isl_printer *print_for_with_pipeline(
  __isl_keep isl_ast_node *node, __isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "#pragma HLS PIPELINE II=1");
  p = isl_printer_end_line(p);

  p = isl_ast_node_for_print(node, p, print_options);

  return p;
}

static __isl_give isl_printer *print_for_with_unroll(
  __isl_keep isl_ast_node *node, __isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "#pragma HLS UNROLL");
  p = isl_printer_end_line(p);

  p = isl_ast_node_for_print(node, p, print_options);

  return p;
}

static __isl_give isl_printer *print_for_xilinx(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  int pipeline;
  int unroll;

  pipeline = 0;
  unroll = 0;
  id = isl_ast_node_get_annotation(node);

  if (id) {
    struct polysa_ast_node_userinfo *info;

    info = (struct polysa_ast_node_userinfo *)isl_id_get_user(id);
    if (info && info->is_pipeline)
      pipeline = 1;
    if (info && info->is_unroll)
      unroll = 1;    
  }

  if (pipeline) 
    p = print_for_with_pipeline(node, p, print_options);
  else if (unroll)
    p = print_for_with_unroll(node, p, print_options);
  else 
    p = isl_ast_node_for_print(node, p, print_options);

  isl_id_free(id);

  return p;
}

static __isl_give isl_printer *print_module_stmt(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  struct polysa_kernel_stmt *stmt;
  struct print_hw_module_data *hw_data = (struct print_hw_module_data *)(user);
  struct polysa_hw_module *module = hw_data->module;

  id = isl_ast_node_get_annotation(node);
  stmt = isl_id_get_user(id);
  isl_id_free(id);

  isl_ast_print_options_free(print_options);
  
  switch (stmt->type) {
//    case POLYSA_KERNEL_STMT_COPY:
//      return polysa_kernel_print_copy(p, stmt);
//    case POLYSA_KERNEL_STMT_SYNC:
//      return print_sync(p, stmt);
    case POLYSA_KERNEL_STMT_DOMAIN:
      return polysa_kernel_print_domain(p, stmt);
    case POLYSA_KERNEL_STMT_IO:
      return polysa_kernel_print_io(p, stmt, hw_data->hls);
    case POLYSA_KERNEL_STMT_IO_TRANSFER:
      return polysa_kernel_print_io_transfer(p, stmt, hw_data->hls);
    case POLYSA_KERNEL_STMT_IO_DRAM:
      return polysa_kernel_print_io_dram(p, stmt, hw_data->hls);
    case POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_TRANS:
      return polysa_kernel_print_inter_trans(p, stmt, hw_data->hls); 
    case POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_TRANS:
      return polysa_kernel_print_intra_trans(p, stmt, hw_data->hls); 
    case POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_INTRA:
      return polysa_kernel_print_inter_intra(p, stmt, hw_data->hls);
    case POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_INTER:
      return polysa_kernel_print_intra_inter(p, stmt, hw_data->hls);
    case POLYSA_KERNEL_STMT_IO_MODULE_CALL_STATE_HANDLE:
      return polysa_kernel_print_state_handle(p, stmt, hw_data->hls); 
  }

  return p;
}

static __isl_give isl_printer *print_top_module_call_stmt(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  struct polysa_kernel_stmt *stmt;
  struct print_hw_module_data *data = (struct print_hw_module_data *)(user); 

  id = isl_ast_node_get_annotation(node);
  stmt = isl_id_get_user(id);
  isl_id_free(id);

  isl_ast_print_options_free(print_options);

  switch (stmt->type) {
    case POLYSA_KERNEL_STMT_MODULE_CALL:
      return polysa_kernel_print_module_call(p, stmt, data->prog);
  }

  return p;
}

static __isl_give isl_printer *print_top_module_fifo_stmt(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  struct polysa_kernel_stmt *stmt;
  struct print_hw_module_data *data = (struct print_hw_module_data *)(user); 

  id = isl_ast_node_get_annotation(node);
  stmt = isl_id_get_user(id);
  isl_id_free(id);

  isl_ast_print_options_free(print_options);

  switch (stmt->type) {
    case POLYSA_KERNEL_STMT_FIFO_DECL:
      return polysa_kernel_print_fifo_decl(p, stmt, data->prog, data->hls);
  }

  return p;
}

/* Print the header of the given module.
 */
static __isl_give isl_printer *print_module_header_xilinx(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_hw_module *module, int inter, int boundary)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "void ");
  p = isl_printer_print_str(p, module->name);
  if (inter == 0)
    p = isl_printer_print_str(p, "_intra_trans");
  else if (inter == 1)
    p = isl_printer_print_str(p, "_inter_trans");
  if (boundary)
    p = isl_printer_print_str(p, "_boundary");
  p = isl_printer_print_str(p, "(");
  p = print_module_arguments(p, prog, module->kernel, module, 1, XILINX_HW, inter, -1, boundary);
  p = isl_printer_print_str(p, ")");

  return p;
}

static __isl_give isl_printer *print_pe_dummy_module_header_xilinx(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_pe_dummy_module *module)
{
  struct polysa_array_ref_group *group = module->io_group;

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "void ");
  // group_name
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
  p = isl_printer_print_str(p, "_PE_dummy");
  p = isl_printer_print_str(p, "(");
  p = print_pe_dummy_module_arguments(p, prog, module->module->kernel, module, 1, XILINX_HW);
  p = isl_printer_print_str(p, ")");

  return p;
}

/* Print the header of the given module.
 */
static __isl_give isl_printer *print_module_header_intel(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_hw_module *module, int inter, int boundary)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "__kernel void ");
  p = isl_printer_print_str(p, module->name);
  if (inter == 0)
    p = isl_printer_print_str(p, "_intra_trans");
  else
    p = isl_printer_print_str(p, "_inter_trans");
  if (boundary)
    p = isl_printer_print_str(p, "boundary");
  p = isl_printer_print_str(p, "(");
  p = print_module_arguments(p, prog, module->kernel, module, 1, INTEL_HW, inter, -1, boundary);
  p = isl_printer_print_str(p, ")");

  return p;
}

/* Extract the data pack factors for each I/O buffer allocated for the current
 * I/O group.
 * Only insert the data pack factor that is not found in the current list
 * "data_pack_factors".
 * The list is in ascending order.
 */
static int *extract_data_pack_factors(int *data_pack_factors,
  int *n_factor, struct polysa_array_ref_group *group)
{
  for (int i = 0; i < group->n_io_buffer; i++) {
    struct polysa_io_buffer *buf = group->io_buffers[i];
    bool insert = true;
    int pos = 0;
    for (pos = 0; pos < *n_factor; pos++) {
      if (buf->n_lane > data_pack_factors[pos]) {
        if (pos < *n_factor - 1) {
          if (buf->n_lane < data_pack_factors[pos + 1]) {
            // insert @pos+1
            pos++;
            break;
          }
        }
      } else if (buf->n_lane == data_pack_factors[pos]) {
        insert = false;
        break;
      }
    }

    if (!insert) 
      continue;

    *n_factor = *n_factor + 1;
    data_pack_factors = (int *)realloc(data_pack_factors, 
          sizeof(int) * (*n_factor));
    for (int j = *n_factor - 1; j > pos; j--) {
      data_pack_factors[j] = data_pack_factors[j - 1];
    }
    data_pack_factors[pos] = buf->n_lane;
  }

  return data_pack_factors;
}

/* Examine the local buffers of each array group. 
 * Extract the data pack factors and build the data types 
 * required by the program. 
 */
static isl_stat *print_data_types_xilinx(
  struct polysa_hw_top_module *top, struct hls_info *hls)
{
  isl_printer *p;
  struct polysa_kernel *kernel;

  kernel = top->kernel;
  p = isl_printer_to_file(kernel->ctx, hls->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  for (int i = 0; i < kernel->n_array; i++) {
    struct polysa_local_array_info *local = &kernel->array[i]; 
    int *data_pack_factors = NULL;
    int n_factor = 0;
    /* IO group */
    for (int n = 0; n < local->n_io_group; n++) {
      struct polysa_array_ref_group *group = local->io_groups[n];
      data_pack_factors = extract_data_pack_factors(data_pack_factors, &n_factor, group);
    }
    /* Drain group */
    if (local->drain_group)
      data_pack_factors = extract_data_pack_factors(data_pack_factors, &n_factor, local->drain_group);

    for (int n = 0; n < n_factor; n++) {
      if (data_pack_factors[n] != 1) {
        int width;

        width = local->array->size * 8 * data_pack_factors[n];
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "typedef ap_uint<");
        p = isl_printer_print_int(p, width);
        p = isl_printer_print_str(p, "> ");
        p = isl_printer_print_str(p, local->array->name);
        p = isl_printer_print_str(p, "_t");
        p = isl_printer_print_int(p, data_pack_factors[n]);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      }
    }
    free(data_pack_factors);
  }

  isl_printer_free(p);

  return isl_stat_ok;
}

/* For Intel devices, we use the vectorized data types */
static isl_stat *print_data_types_intel(
  struct polysa_hw_top_module *top, struct hls_info *hls)
{
  isl_printer *p;
  struct polysa_kernel *kernel;

  kernel = top->kernel;
  p = isl_printer_to_file(kernel->ctx, hls->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  for (int i = 0; i < kernel->n_array; i++) {
    struct polysa_local_array_info *local = &kernel->array[i]; 
    // TODO: add SIMD later
    int *data_pack_factors = NULL;
    int n_factor = 0;
    /* IO group */
    for (int n = 0; n < local->n_io_group; n++) {
      struct polysa_array_ref_group *group = local->io_groups[n];
      data_pack_factors = extract_data_pack_factors(data_pack_factors, &n_factor, group);
    }
    /* Drain group */
    if (local->drain_group)
      data_pack_factors = extract_data_pack_factors(data_pack_factors, &n_factor, local->drain_group);

    for (int n = 0; n < n_factor; n++) {
      if (data_pack_factors[n] != 1) {
        p = isl_printer_start_line(p);
        p = isl_printer_print_str(p, "typedef ");
        p = isl_printer_print_str(p, local->array->type);
        p = isl_printer_print_int(p, data_pack_factors[n]);
        p = isl_printer_print_str(p, " ");
        p = isl_printer_print_str(p, local->array->name);
        p = isl_printer_print_str(p, "_t");
        p = isl_printer_print_int(p, data_pack_factors[n]);
        p = isl_printer_print_str(p, ";");
        p = isl_printer_end_line(p);
      }
    }
  }

  isl_printer_free(p);
 
  return isl_stat_ok;
}

static isl_stat print_pe_dummy_module_headers_xilinx(
  struct polysa_prog *prog, struct polysa_pe_dummy_module *module, struct hls_info *hls)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, hls->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_pe_dummy_module_header_xilinx(p, prog, module);
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  p = isl_printer_to_file(prog->ctx, hls->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_pe_dummy_module_header_xilinx(p, prog, module);
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  return isl_stat_ok;
}

/* Print the header of the given module to both gen->hls.kernel_h
 * and gen->hls.kernel_c
 * If "inter" is -1, this is a normal module call.
 * If "inter" is 0, this is a intra_trans module call.
 * If "inter" is 1, this is a inter_trans module call.
 */
static isl_stat print_module_headers_xilinx(
  struct polysa_prog *prog, struct polysa_hw_module *module, struct hls_info *hls, int inter, int boundary)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, hls->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_module_header_xilinx(p, prog, module, inter, boundary); 
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  p = isl_printer_to_file(prog->ctx, hls->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_module_header_xilinx(p, prog, module, inter, boundary);
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  return isl_stat_ok;
}

/* Print the header of the given module to both gen->hls.kernel_h
 * and gen->hls.kernel_c
 * If "inter" is -1, this is a normal module call.
 * If "inter" is 0, this is a intra_trans module call.
 * If "inter" is 1, this is a inter_trans module call. 
 */
isl_stat print_module_headers_intel(
  struct polysa_prog *prog, struct polysa_hw_module *module, struct hls_info *hls, int inter, int boundary)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, hls->kernel_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_module_header_intel(p, prog, module, inter, boundary); 
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  p = isl_printer_to_file(prog->ctx, hls->kernel_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "__attribute__((max_global_work_dim(0)))");
  p = isl_printer_end_line(p);
  p = print_module_header_intel(p, prog, module, inter, boundary);
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  return isl_stat_ok;
}

static void print_module_iterators(FILE *out, struct polysa_hw_module *module)
{
  isl_ctx *ctx;
  const char *type;
  const char *dims[] = { "idx", "idy", "idz" };

  ctx = isl_ast_node_get_ctx(module->tree);
  type = isl_options_get_ast_iterator_type(ctx);

  ctx = isl_ast_node_get_ctx(module->device_tree);
//  // debug
//  isl_printer *p = isl_printer_to_file(ctx, stdout);
//  p = isl_printer_print_isl_id_list(p, module->inst_ids);
//  printf("\n");
//  // debug
  print_iterators(out, type, module->inst_ids, dims);
}

static __isl_give isl_printer *print_module_vars_xilinx(__isl_take isl_printer *p,
  struct polysa_hw_module *module, int inter)
{
  int i, n;
  isl_space *space;  
  const char *type;

//  /* Print the tmp register for I/O module. */
//  if (module->type == IO_MODULE) {
//    p = isl_printer_start_line(p);
//    p = isl_printer_print_str(p, module->io_groups[0]->array->type) ;
//    p = isl_printer_print_str(p, " ");
//    p = isl_printer_print_str(p, "fifo_data;");
//    p = isl_printer_end_line(p);
//  }

  if (inter == -1) {
    for (i = 0; i < module->n_var; ++i)
      p = print_kernel_var_xilinx(p, &module->var[i], module->double_buffer);
  }

  if (module->double_buffer && inter == -1) {
    type = isl_options_get_ast_iterator_type(module->kernel->ctx);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "bool arb = 0;");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, module->in? "bool inter_trans_en = 1;" :
        "bool inter_trans_en = 0;");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, module->in? "bool intra_trans_en = 0;" :
        "bool intra_trans_en = 1;");
    p = isl_printer_end_line(p);
    /* iterators */
    space = (module->in)? module->intra_space : module->inter_space;
    n = isl_space_dim(space, isl_dim_set);
    for (int i = 0; i < n; i++) {
      const char *name;
      name = isl_space_get_dim_name(space, isl_dim_set, i);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
      p = isl_printer_print_str(p, name);
      p = isl_printer_print_str(p, ", ");
      p = isl_printer_print_str(p, name);
      p = isl_printer_print_str(p, "_prev");
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
    }
  }

  return p;
}

static __isl_give isl_printer *print_module_vars_intel(__isl_take isl_printer *p,
  struct polysa_hw_module *module, int inter)
{
  int i, n;
  isl_space *space;  
  const char *type;

//  /* Print the tmp register for I/O module. */
//  if (module->type == IO_MODULE) {
//    p = isl_printer_start_line(p);
//    p = isl_printer_print_str(p, module->io_groups[0]->array->type) ;
//    p = isl_printer_print_str(p, " ");
//    p = isl_printer_print_str(p, "fifo_data;");
//    p = isl_printer_end_line(p);
//  }

  if (inter == -1) {
    for (i = 0; i < module->n_var; ++i)
      p = print_kernel_var_intel(p, &module->var[i], module->double_buffer);
  }

  if (module->double_buffer && inter == -1) {
    type = isl_options_get_ast_iterator_type(module->kernel->ctx);

    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "bool arb = 0;");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, module->in? "bool inter_trans_en = 1;" :
        "bool inter_trans_en = 0;");
    p = isl_printer_end_line(p);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, module->in? "bool intra_trans_en = 0;" :
        "bool intra_trans_en = 1;");
    p = isl_printer_end_line(p);
    /* iterators */
    space = (module->in)? module->intra_space : module->inter_space;
    n = isl_space_dim(space, isl_dim_set);
    for (int i = 0; i < n; i++) {
      const char *name;
      name = isl_space_get_dim_name(space, isl_dim_set, i);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, type);
      p = isl_printer_print_str(p, " ");
      p = isl_printer_print_str(p, name);
      p = isl_printer_print_str(p, ";");
      p = isl_printer_end_line(p);
    }
  }

  return p;
}

static __isl_give isl_printer *print_top_gen_header(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_hw_top_module *top)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "void ");
  p = isl_printer_print_str(p, "top_generate");
  p = isl_printer_print_str(p, "(");
  p = print_top_gen_arguments(p, prog, top->kernel, 1); 
  p = isl_printer_print_str(p, ")");

  return p;
}

static __isl_give isl_printer *print_top_gen_headers(
  struct polysa_prog *prog, struct polysa_hw_top_module *top, struct hls_info *hls)
{
  isl_printer *p;

  p = isl_printer_to_file(prog->ctx, hls->top_gen_h);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_top_gen_header(p, prog, top); 
  p = isl_printer_print_str(p, ";");
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  p = isl_printer_to_file(prog->ctx, hls->top_gen_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = print_top_gen_header(p, prog, top); 
  p = isl_printer_end_line(p);
  isl_printer_free(p);
}

static __isl_give isl_printer *print_top_module_headers(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_hw_top_module *top, struct hls_info *hls)
{
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"void kernel");
  p = isl_printer_print_int(p, top->kernel->id);
  p = isl_printer_print_str(p, "(");
  p = print_kernel_arguments(p, prog, top->kernel, 1);
  p = isl_printer_print_str(p, ")\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"{\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *print_top_module_interface_xilinx(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_kernel *kernel)
{
  int n;
  unsigned nparam;
  isl_space *space;
  const char *type;

  for (int i = 0; i < prog->n_array; ++i) {
    struct polysa_array_info *array = &prog->array[i];
    if (polysa_kernel_requires_array_argument(kernel, i) && !polysa_array_is_scalar(array)) {
      p = print_str_new_line(p, "p = isl_printer_start_line(p);");
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"#pragma HLS INTERFACE m_axi port=");
      p = isl_printer_print_str(p, array->name);
      p = isl_printer_print_str(p, " offset=slave bundle=gmem_");
      p = isl_printer_print_str(p, array->name);
      p = isl_printer_print_str(p, "\");");
      p = isl_printer_end_line(p);
      p = print_str_new_line(p, "p = isl_printer_end_line(p);");
    }
  }

  for (int i = 0; i < prog->n_array; ++i) {
    struct polysa_array_info *array = &prog->array[i];
    if (polysa_kernel_requires_array_argument(kernel, i)) {
      p = print_str_new_line(p, "p = isl_printer_start_line(p);");
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"#pragma HLS INTERFACE s_axilite port=");
      p = isl_printer_print_str(p, array->name);
      p = isl_printer_print_str(p, " bundle=control\");");
      p = isl_printer_end_line(p);
      p = print_str_new_line(p, "p = isl_printer_end_line(p);");
    }
  }

  space = isl_union_set_get_space(kernel->arrays);
  nparam = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < nparam; i++) {
    const char *name;
    name = isl_space_get_dim_name(space, isl_dim_param, i);
    p = print_str_new_line(p, "p = isl_printer_start_line(p);");
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"#pragma HLS INTERFACE s_axilite port=");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, " bundle=control\");");
    p = isl_printer_end_line(p);
    p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  }
  isl_space_free(space);

  n = isl_space_dim(kernel->space, isl_dim_set);
  type = isl_options_get_ast_iterator_type(prog->ctx);
  for (int i = 0; i < n; i++) {
    const char *name;
    name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
    p = print_str_new_line(p, "p = isl_printer_start_line(p);");
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"#pragma HLS INTERFACE s_axilite port=");
    p = isl_printer_print_str(p, name);
    p = isl_printer_print_str(p, " bundle=control\");");
    p = isl_printer_end_line(p);
    p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  }

  p = print_str_new_line(p, "p = isl_printer_start_line(p);");
  p = print_str_new_line(p, "p = isl_printer_print_str(p, \"#pragma HLS INTERFACE s_axilite port=return bundle=control\");");
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");

  return p;
}

static __isl_give isl_printer *print_top_module_headers_xilinx(__isl_take isl_printer *p,
  struct polysa_prog *prog, struct polysa_hw_top_module *top, struct hls_info *hls)
{
  struct polysa_kernel *kernel = top->kernel;

  p = print_str_new_line(p, "p = isl_printer_start_line(p);");
  p = print_str_new_line(p, "p = isl_printer_print_str(p, \"extern \\\"C\\\" {\");");
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");

  p = print_str_new_line(p, "p = isl_printer_start_line(p);");

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"void kernel");
  p = isl_printer_print_int(p, top->kernel->id);
  p = isl_printer_print_str(p, "(");
  p = print_kernel_arguments(p, prog, top->kernel, 1);
  p = isl_printer_print_str(p, ")\");");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  p = print_str_new_line(p, "p = isl_printer_start_line(p);");
  p = print_str_new_line(p, "p = isl_printer_print_str(p, \"{\");");
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");

  /* Print out the interface pragmas */
  p = print_top_module_interface_xilinx(p, prog, kernel);

  /* Print out the dataflow pragma */
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  p = print_str_new_line(p, "p = isl_printer_start_line(p);");
  p = print_str_new_line(p, "p = isl_printer_print_str(p, \"#pragma HLS DATAFLOW\");");
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");

  p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  
  return p;
}

static char *extract_fifo_name_from_fifo_decl_name(isl_ctx *ctx, char *fifo_decl_name) {
  int loc = 0;
  char ch;
  isl_printer *p_str = isl_printer_to_str(ctx);
  char *name = NULL;

  while ((ch = fifo_decl_name[loc]) != '\0') {
    if (ch == '.')
      break;
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  return name;
}

static char *extract_fifo_width_from_fifo_decl_name(isl_ctx *ctx, char *fifo_decl_name) {
  int loc = 0;
  char ch;
  isl_printer *p_str = isl_printer_to_str(ctx);
  char *name = NULL;

  while ((ch = fifo_decl_name[loc]) != '\0') {
    if (ch == '.')
      break;
    loc++;
  }

  loc++;

  while ((ch = fifo_decl_name[loc]) != '\0') {
    char buf[2];
    buf[0] = ch;
    buf[1] = '\0';
    p_str = isl_printer_print_str(p_str, buf);
    loc++;
  }

  name = isl_printer_get_str(p_str);
  isl_printer_free(p_str);

  return name;
}

static void print_top_gen_host_code(
  struct polysa_prog *prog, __isl_keep isl_ast_node *node,
  struct polysa_hw_top_module *top, struct hls_info *hls)
{
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_ast_node_get_ctx(node);
  isl_printer *p;
  struct print_hw_module_data hw_data = { hls, prog, NULL };

  /* Print the top module ASTs. */
  p = isl_printer_to_file(ctx, hls->top_gen_c);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);

  print_top_gen_headers(prog, top, hls);
  fprintf(hls->top_gen_c, "{\n");
  p = isl_printer_indent(p, 4);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "FILE *fd = fopen(\"design_info.dat\", \"w\");");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "int fifo_cnt;");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "isl_ctx *ctx = isl_ctx_alloc();");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "isl_printer *p = isl_printer_to_file(ctx, f);");
  p = isl_printer_end_line(p);
  p = isl_printer_end_line(p);

  if (hls->target == XILINX_HW)
    p = print_top_module_headers_xilinx(p, prog, top, hls);
  else
    p = print_top_module_headers(p, prog, top, hls);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_indent(p, 4);");
  p = isl_printer_end_line(p);

  /* Print FIFO declarations */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* FIFO Declaration */\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);
  p = isl_printer_end_line(p);

  for (int i = 0; i < top->n_fifo_decls; i++) {
    /* Generate fifo decl counter */
    char *fifo_decl_name = top->fifo_decl_names[i];
    char *fifo_name = extract_fifo_name_from_fifo_decl_name(ctx, fifo_decl_name);
    char *fifo_w = extract_fifo_width_from_fifo_decl_name(ctx, fifo_decl_name);
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fifo_cnt = 0;");
    p = isl_printer_end_line(p);

    /* Print AST */
    print_options = isl_ast_print_options_alloc(ctx);
    print_options = isl_ast_print_options_set_print_user(print_options,
                      &print_top_module_fifo_stmt, &hw_data); 
  
    p = isl_ast_node_print(top->fifo_decl_wrapped_trees[i], 
          p, print_options); 

    /* fifo:fifo_name:fifo_cnt:fifo_width */
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fprintf(fd, \"fifo:");
    p = isl_printer_print_str(p, fifo_name);
    p = isl_printer_print_str(p, ":\%d:");
    p = isl_printer_print_str(p, fifo_w);
    p = isl_printer_print_str(p, "\\n\", fifo_cnt);");
    p = isl_printer_end_line(p);

    p = isl_printer_end_line(p);

    free(fifo_name);
    free(fifo_w);
  }

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_start_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_print_str(p, \"/* FIFO Declaration */\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_end_line(p);");
  p = isl_printer_end_line(p);

  int n_module_names = 0;
  char **module_names = NULL;
  for (int i = 0; i < top->n_hw_modules; i++) {
    /* Generate module call counter */
    struct polysa_hw_module *module = top->hw_modules[i];
    char *module_name;

    if (module->is_filter && module->is_buffer) {
      module_name = concat(ctx, module->name, "intra_trans");
      
      n_module_names++;
      module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
      module_names[n_module_names - 1] = module_name;

      module_name = concat(ctx, module->name, "inter_trans");
      
      n_module_names++;
      module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
      module_names[n_module_names - 1] = module_name;

      if (module->boundary) {
        module_name = concat(ctx, module->name, "inter_trans_boundary");
      
        n_module_names++;
        module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
        module_names[n_module_names - 1] = module_name;       
      }
    }
        
    module_name = strdup(module->name);
    
    n_module_names++;
    module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
    module_names[n_module_names - 1] = module_name;       
 
    if (module->boundary) {
      module_name = concat(ctx, module->name, "boundary");
      
      n_module_names++;
      module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
      module_names[n_module_names - 1] = module_name;       
    }

    if (module->n_pe_dummy_modules > 0) {
      for (int j = 0; j < module->n_pe_dummy_modules; j++) {
        struct polysa_pe_dummy_module *dummy_module = module->pe_dummy_modules[j];
        struct polysa_array_ref_group *group = dummy_module->io_group;
        isl_printer *p_str = isl_printer_to_str(ctx);
        p_str = polysa_array_ref_group_print_prefix(group, p_str);
        p_str = isl_printer_print_str(p_str, "_PE_dummy");
        module_name = isl_printer_get_str(p_str);
        isl_printer_free(p_str);
        
        n_module_names++;
        module_names = (char **)realloc(module_names, n_module_names * sizeof(char *));
        module_names[n_module_names - 1] = module_name;       
      }
    }
  }
  for (int i = 0; i < n_module_names; i++) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "int ");
    p = isl_printer_print_str(p, module_names[i]);
    p = isl_printer_print_str(p, "_cnt = 0;");
    p = isl_printer_end_line(p);
  }

  /* Print module calls */
  for (int i = 0; i < top->n_module_calls; i++) {
    /* Print AST */
    print_options = isl_ast_print_options_alloc(ctx);
    print_options = isl_ast_print_options_set_print_user(print_options,
                      &print_top_module_call_stmt, &hw_data); 
  
    p = isl_ast_node_print(top->module_call_wrapped_trees[i], 
          p, print_options);    
  }

  // TODO: fix inter/intra trans
  /* module:module_name:module_cnt */
  for (int i = 0; i < n_module_names; i++) {
    p = isl_printer_start_line(p);
    p = isl_printer_print_str(p, "fprintf(fd, \"module:");
    p = isl_printer_print_str(p, module_names[i]);
    p = isl_printer_print_str(p, ":\%d\\n\", ");
    p = isl_printer_print_str(p, module_names[i]);
    p = isl_printer_print_str(p, "_cnt);");
    p = isl_printer_end_line(p);
  }
  p = isl_printer_end_line(p);   

  for (int i = 0; i < n_module_names; i++) {
    free(module_names[i]);
  }
  free(module_names);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "p = isl_printer_indent(p, -4);");
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "p = isl_printer_start_line(p);");
  p = print_str_new_line(p, "p = isl_printer_print_str(p, \"}\");");
  p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  if (hls->target == XILINX_HW) { 
    p = print_str_new_line(p, "p = isl_printer_start_line(p);");
    p = print_str_new_line(p, "p = isl_printer_print_str(p, \"}\");");
    p = print_str_new_line(p, "p = isl_printer_end_line(p);");
  }

  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "fclose(fd);");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "isl_printer_free(p);");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "isl_ctx_free(ctx);");
  p = isl_printer_end_line(p);
  p = isl_printer_indent(p, -4);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "}");
  p = isl_printer_end_line(p);
  p = isl_printer_end_line(p);

  /* For internal testing only */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "int main()");
  p = isl_printer_end_line(p);

  p = ppcg_start_block(p);
  
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "FILE *f = fopen(\"temp.c\", \"w\");");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "top_generate(f);");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p);
  p = isl_printer_free(p);
  
  return;
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
static __isl_give isl_printer *print_host_user_xilinx(__isl_take isl_printer *p,
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
    return print_device_node_xilinx(p, node, data->prog); 

  is_user = !strcmp(isl_id_get_name(id), "user");
  kernel = is_user ? NULL : isl_id_get_user(id);
  stmt = is_user ? isl_id_get_user(id) : NULL;
  isl_id_free(id);

  if (is_user)
    return polysa_kernel_print_domain(p, stmt); 

  p = ppcg_start_block(p); 

  p = print_set_kernel_arguments_xilinx(p, data->prog, kernel);

  p = print_str_new_line(p, "// Launch the Kernel");
  p = print_str_new_line(p, "OCL_CHECK(err, err = q.enqueueTask(krnl));");
  p = isl_printer_end_line(p);

  /* Print the top kernel generation function */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Top Function Generation */");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "FILE *f = fopen(\"top.c\", \"w\");");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "top_generate(");
  p = print_top_gen_arguments(p, data->prog, kernel, 0);
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "fclose(f);");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Top Function Generation */");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  /* Print the top kernel header */
  print_kernel_headers_xilinx(data->prog, kernel, data->hls);

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
static __isl_give isl_printer *print_host_user_intel(__isl_take isl_printer *p,
  __isl_take isl_ast_print_options *print_options,
  __isl_keep isl_ast_node *node, void *user)
{
  isl_id *id;
  int is_user;
  struct polysa_kernel *kernel;
  struct polysa_kernel_stmt *stmt;
  struct print_host_user_data *data;
  int indent;
  struct polysa_hw_top_module *top;

  isl_ast_print_options_free(print_options);

  data = (struct print_host_user_data *) user;

  id = isl_ast_node_get_annotation(node);
  if (!id)
    return print_device_node_intel(p, node, data->prog, data->top); 

  is_user = !strcmp(isl_id_get_name(id), "user");
  kernel = is_user ? NULL : isl_id_get_user(id);
  stmt = is_user ? isl_id_get_user(id) : NULL;
  isl_id_free(id);

  if (is_user)
    return polysa_kernel_print_domain(p, stmt); 

  p = ppcg_start_block(p); 
  top = data->top;

  for (int i = 0; i < top->n_hw_modules; i++) {
    struct polysa_hw_module *module = top->hw_modules[i];
    struct polysa_array_ref_group *group = module->io_groups[0];
    if (module->type != PE_MODULE && module->to_mem) {            
      p = print_str_new_line(p, "status = clSetKernelArg(");
      p = isl_printer_indent(p, 4);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "kernel[KID_");
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
      p = isl_printer_print_str(p, "],");
      p = isl_printer_end_line(p);
      p = print_str_new_line(p, "0,");
      p = print_str_new_line(p, "sizeof(cl_mem),");
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "(void*)&buf_");
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "); CHECK(status);");
      p = isl_printer_end_line(p);
      p = isl_printer_indent(p, -4);
    }
  }
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "size_t globalWorkSize[1];");
  p = print_str_new_line(p, "size_t localWorkSize[1];");
  p = print_str_new_line(p, "globalWorkSize[0] = 1;");
  p = print_str_new_line(p, "localWorkSize[0] = 1;");
  p = isl_printer_end_line(p);
  
  p = print_str_new_line(p, "// Enqueue kernels");
  for (int i = 0; i < top->n_hw_modules; i++) {
    struct polysa_hw_module *module = top->hw_modules[i];
    struct polysa_array_ref_group *group = module->io_groups[0];
    if (module->type != PE_MODULE && module->to_mem) {
      p = print_str_new_line(p, "status = clEnqueueNDRangeKernel(");
      p = isl_printer_indent(p, 4);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "cmdQueue[QID_");
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "],");
      p = isl_printer_end_line(p);
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "kernel[KID_");
      p = isl_printer_print_str(p, group->array->name);
      p = isl_printer_print_str(p, "_");
      p = isl_printer_print_int(p, group->nr);
      p = isl_printer_print_str(p, "],");
      p = isl_printer_end_line(p);
      p = print_str_new_line(p, "1,");
      p = print_str_new_line(p, "NULL,");
      p = print_str_new_line(p, "globalWorkSize,");
      p = print_str_new_line(p, "localWorkSize,");
      p = print_str_new_line(p, "0,");
      p = print_str_new_line(p, "NULL,");
      p = print_str_new_line(p, "NULL); CHECK(statis);");
      p = isl_printer_indent(p, -4);
    }
  }
  p = isl_printer_end_line(p);

  p = print_str_new_line(p, "for (int i = 0; i < NUM_QUEUES_TO_CREATE; i++) {");
  p = isl_printer_indent(p, 4);
  p = print_str_new_line(p, "status = clFinish(cmdQueue[i]); CHECK(status);");
  p = isl_printer_indent(p, -4);
  p = print_str_new_line(p, "}");
  p = isl_printer_end_line(p);

  /* Print the top kernel generation function */
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Top Function Generation */");
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "FILE *f = fopen(\"top.c\", \"w\");");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "top_generate(");
  p = print_top_gen_arguments(p, data->prog, kernel, 0);
  p = isl_printer_print_str(p, ");");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "fclose(f);");
  p = isl_printer_end_line(p);
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Top Function Generation */");
  p = isl_printer_end_line(p);

  p = ppcg_end_block(p); 

  p = isl_printer_start_line(p);
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *polysa_print_default_pe_dummy_module(__isl_take isl_printer *p,
  struct polysa_pe_dummy_module *pe_dummy_module,
  struct polysa_prog *prog, struct hls_info *hls, int boundary)
{
  struct polysa_hw_module *module = pe_dummy_module->module;
  struct print_hw_module_data hw_data = {hls, prog, module};
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_printer_get_ctx(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);

  if (hls->target == XILINX_HW)
    print_pe_dummy_module_headers_xilinx(prog, pe_dummy_module, hls);
  else {
    // TODO
  }
  
  fprintf(hls->kernel_c, "{\n");
  print_module_iterators(hls->kernel_c, module);
  
  p = isl_printer_indent(p, 4);
//  if (hls->target == XILINX_HW)
//    p = print_module_vars_xilinx(p, module, -1);
//  else if (hls->target == INTEL_HW)
//    p = print_module_vars_intel(p, module, -1);
  p = isl_printer_end_line(p);
  
  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                    &print_module_stmt, &hw_data); 
  if (hls->target == XILINX_HW) {
    print_options = isl_ast_print_options_set_print_for(print_options,
                      &print_for_xilinx, &hw_data);
  }
  
  p = isl_ast_node_print(pe_dummy_module->device_tree, p, print_options);
  
  p = isl_printer_indent(p, -4);
   
  fprintf(hls->kernel_c, "}\n");
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  p = isl_printer_end_line(p);

  return p;

}

static __isl_give isl_printer *polysa_print_default_module(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_prog *prog,
  struct hls_info *hls, int boundary)
{
  struct print_hw_module_data hw_data = {hls, prog, module};
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_printer_get_ctx(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);

  if (hls->target == XILINX_HW)
    print_module_headers_xilinx(prog, module, hls, -1, boundary);
  else
    print_module_headers_intel(prog, module, hls, -1, boundary);
  
  fprintf(hls->kernel_c, "{\n");
  print_module_iterators(hls->kernel_c, module);
  
  p = isl_printer_indent(p, 4);
  if (hls->target == XILINX_HW)
    p = print_module_vars_xilinx(p, module, -1);
  else if (hls->target == INTEL_HW)
    p = print_module_vars_intel(p, module, -1);
  p = isl_printer_end_line(p);
  
  if (module->credit && !module->in) {
    if (hls->target == XILINX_HW) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "credit.write(1);");
      p = isl_printer_end_line(p);
    } else if (hls->target == INTEL_HW) {
      // TODO
    }
  }
  
  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                    &print_module_stmt, &hw_data); 
  if (hls->target == XILINX_HW) {
    print_options = isl_ast_print_options_set_print_for(print_options,
                      &print_for_xilinx, &hw_data);
  }
  
  if (!boundary)
    p = isl_ast_node_print(module->device_tree, p, print_options);
  else
    p = isl_ast_node_print(module->boundary_tree, p, print_options);
  
  if (module->credit && module->in) {
    if (hls->target == XILINX_HW) {
      p = isl_printer_start_line(p);
      p = isl_printer_print_str(p, "int token = credit.read();");
      p = isl_printer_end_line(p);
    } else if (hls->target == INTEL_HW) {
      // TODO
    }
  }
  
  p = isl_printer_indent(p, -4);
   
  fprintf(hls->kernel_c, "}\n");
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  p = isl_printer_end_line(p);

  return p;
}

__isl_give isl_printer *polysa_print_intra_trans_module(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_prog *prog,
  struct hls_info *hls, int boundary)
{
  struct print_hw_module_data hw_data = {hls, prog, module};
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_printer_get_ctx(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  if (hls->target == XILINX_HW)
    print_module_headers_xilinx(prog, module, hls, 0, boundary); 
  if (hls->target == INTEL_HW)
    print_module_headers_intel(prog, module, hls, 0, boundary);
  fprintf(hls->kernel_c, "{\n");
  if (hls->target == XILINX_HW)
    fprintf(hls->kernel_c, "#pragma HLS INLINE OFF\n");
  print_module_iterators(hls->kernel_c, module);
  
  p = isl_printer_indent(p, 4);
  if (hls->target == XILINX_HW)
    p = print_module_vars_xilinx(p, module, 0);
  else if (hls->target == INTEL_HW)
    p = print_module_vars_intel(p, module, 0);
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "if (!intra_trans_en) return;");
  p = isl_printer_end_line(p);
  p = isl_printer_end_line(p);

  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                    &print_module_stmt, &hw_data); 
  if (hls->target == XILINX_HW) {
    print_options = isl_ast_print_options_set_print_for(print_options,
                      &print_for_xilinx, &hw_data);
  }

  p = isl_ast_node_print(module->intra_tree, p, print_options);
  p = isl_printer_indent(p, -4);

  fprintf(hls->kernel_c, "}\n");
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *polysa_print_inter_trans_module(__isl_take isl_printer *p,
  struct polysa_hw_module *module, struct polysa_prog *prog,
  struct hls_info *hls, int boundary)
{
  struct print_hw_module_data hw_data = {hls, prog, module};
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_printer_get_ctx(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  if (hls->target == XILINX_HW)
    print_module_headers_xilinx(prog, module, hls, 1, boundary); 
  if (hls->target == INTEL_HW)
    print_module_headers_intel(prog, module, hls, 1, boundary);
  fprintf(hls->kernel_c, "{\n");
  fprintf(hls->kernel_c, "#pragma HLS INLINE OFF\n");
  print_module_iterators(hls->kernel_c, module);
  
  p = isl_printer_indent(p, 4);
  if (hls->target == XILINX_HW)
    p = print_module_vars_xilinx(p, module, 1);
  else if (hls->target == INTEL_HW)
    p = print_module_vars_intel(p, module, 1);
  p = isl_printer_end_line(p);

  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "if (!inter_trans_en) return;");
  p = isl_printer_end_line(p);
  p = isl_printer_end_line(p);

  print_options = isl_ast_print_options_alloc(ctx);
  print_options = isl_ast_print_options_set_print_user(print_options,
                    &print_module_stmt, &hw_data); 
  if (hls->target == XILINX_HW) {
    print_options = isl_ast_print_options_set_print_for(print_options,
                      &print_for_xilinx, &hw_data);
  }
 
  p = isl_ast_node_print((boundary == 0)? module->inter_tree : module->boundary_inter_tree,
        p, print_options);
  p = isl_printer_indent(p, -4);

  fprintf(hls->kernel_c, "}\n");
  p = isl_printer_start_line(p);
  p = isl_printer_print_str(p, "/* Module Definition */");
  p = isl_printer_end_line(p);
  
  p = isl_printer_end_line(p);

  return p;
}

static __isl_give isl_printer *polysa_print_host_code(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct polysa_hw_top_module *top,
  struct hls_info *hls)
{
  isl_ast_print_options *print_options;
  isl_ctx *ctx = isl_ast_node_get_ctx(tree);
  struct print_host_user_data data = { hls, prog, top };
  struct print_hw_module_data hw_data = { hls, prog, NULL };
  isl_printer *p_module;

  /* Print the data pack types in the program. */
  if (hls->target == XILINX_HW)
    print_data_types_xilinx(top, hls);
  else
    print_data_types_intel(top, hls);

  /* Print the default AST. */
  print_options = isl_ast_print_options_alloc(ctx);

  if (hls->target == XILINX_HW) {
    print_options = isl_ast_print_options_set_print_user(print_options,
                  &print_host_user_xilinx, &data);
  } else if (hls->target == INTEL_HW) {
    print_options = isl_ast_print_options_set_print_user(print_options,
                  &print_host_user_intel, &data);
  }

  /* Print the macros definitions in the program. */
  p = polysa_print_macros(p, tree); 
  p = isl_ast_node_print(tree, p, print_options);

  /* Print the hw module ASTs. */
  p_module = isl_printer_to_file(ctx, hls->kernel_c);
  p_module = isl_printer_set_output_format(p_module, ISL_FORMAT_C);

  for (int i = 0; i < n_modules; i++) {
    if (modules[i]->is_filter && modules[i]->is_buffer) {
      /* Print out the definitions for inter_trans and intra_trans function calls */
      /* Intra transfer function */
      p_module = polysa_print_intra_trans_module(p_module, modules[i], prog, hls, 0);
       
      /* Inter transfer function */
      p_module = polysa_print_inter_trans_module(p_module, modules[i], prog, hls, 0);
      if (modules[i]->boundary)
        p_module = polysa_print_inter_trans_module(p_module, modules[i], prog, hls, 1);
    }

    p_module = polysa_print_default_module(p_module, modules[i], prog, hls, 0);
    if (modules[i]->boundary) {
      /* Print out the definitions for boundary trans function calls */
      p_module = polysa_print_default_module(p_module, modules[i], prog, hls, 1); 
    }
    if (modules[i]->n_pe_dummy_modules > 0) {
      /* Print out the definitions for pe dummy function calls */
      for (int j = 0; j < modules[i]->n_pe_dummy_modules; j++) {
        p_module = polysa_print_default_pe_dummy_module(
            p_module, modules[i]->pe_dummy_modules[j], prog, hls, 0);
      }
    }
  }
  isl_printer_free(p_module);
    
  return p;
}

/* Given a polysa_prog "prog" and the corresponding tranformed AST
 * "tree", print the entire OpenCL/HLS code to "p".
 * "types" collecs the types for which a definition has already been
 * printed.
 */
static __isl_give isl_printer *print_hw(__isl_take isl_printer *p,
  struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
  struct polysa_hw_module **modules, int n_modules,
  struct polysa_hw_top_module *top_module,
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

  /* Print OpenCL host and kernel function */
  p = polysa_print_host_code(p, prog, tree, modules, n_modules, top_module, hls); 
  /* Print seperate top module code generation function */
  print_top_gen_host_code(prog, tree, top_module, hls); 

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
int generate_polysa_intel_opencl(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input) 
{
  struct hls_info hls;
  int r;

  hls.target = INTEL_HW;
  opencl_open_files(&hls, input);

  r = generate_sa(ctx, input, hls.host_c, options, &print_hw, &hls);

  hls_close_files(&hls);
}

int generate_polysa_xilinx_hls(isl_ctx *ctx, struct ppcg_options *options, 
  const char *input) 
{
  struct hls_info hls;
  int r;

  hls.target = XILINX_HW;
  hls_open_files(&hls, input);

  r = generate_sa(ctx, input, hls.host_c, options, &print_hw, &hls);

  hls_close_files(&hls);
}
