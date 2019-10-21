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

static __isl_give isl_printer *print_t2s(__isl_take isl_printer *p,
		struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		struct gpu_types *types, void *user) {
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
