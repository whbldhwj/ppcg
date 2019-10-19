#include <stdlib.h>
#include <string.h>

#include <isl/polynomial.h>
#include <isl/union_set.h>
#include <isl/aff.h>
#include <isl/ilp.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/options.h>
#include <isl/ast_build.h>

#include "schedule.h"
#include "ppcg_options.h"
#include "print.h"
#include "util.h"
#include "polysa_sa.h"

static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct polysa_gen *gen, struct ppcg_scop *scop,
	struct ppcg_options *options)
{
	struct polysa_prog *prog;
	isl_ctx *ctx;
	isl_schedule *schedule = NULL;
	isl_bool any_permutable;

//  if (!scop)
//		return isl_printer_free(p);
//
//	ctx = isl_printer_get_ctx(p);
//	prog = gpu_prog_alloc(ctx, scop);
//	if (!prog)
//		return isl_printer_free(p);
//
//	gen->prog = prog;
//	schedule = get_schedule(gen);
//
//	any_permutable = has_any_permutable_node(schedule);
//	if (any_permutable < 0 || !any_permutable) {
//		if (any_permutable < 0)
//			p = isl_printer_free(p);
//		else
//			p = print_cpu(p, scop, options);
//		isl_schedule_free(schedule);
//	} else {
////    // debug
////    isl_printer *printer = isl_printer_to_file(isl_schedule_get_ctx(schedule), stdout);
////    isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
////    isl_printer_print_schedule(printer, schedule);
////    printf("\n");
////    // debug
//		schedule = map_to_device(gen, schedule);
////    // debug
////    isl_printer_print_schedule(printer, schedule);
////    printf("\n");
////    isl_printer_free(printer);
////    // debug
//		gen->tree = generate_code(gen, schedule);
//		p = ppcg_set_macro_names(p);
//		p = ppcg_print_exposed_declarations(p, prog->scop);
//		p = gen->print(p, gen->prog, gen->tree, &gen->types,
//				    gen->print_user);
//		isl_ast_node_free(gen->tree);
//	}

//	polysa_prog_free(prog);

	return p;
}


/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *generate_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct polysa_gen *gen = user;

	return generate(p, gen, scop, gen->options);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding systolic code and write the results to "out".
 */
int generate_sa(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*print)(__isl_take isl_printer *p,
		struct polysa_prog *prog, __isl_keep isl_ast_node *tree,
		struct polysa_types *types, void *user), void *user)
{
	struct polysa_gen gen;
	int r;
	int i;

	gen.ctx = ctx;
  // TODO: support later
//	gen.sizes = extract_sizes_from_str(ctx, options->sizes);
	gen.options = options;
	gen.kernel_id = 0;
	gen.print = print;
	gen.print_user = user;
	gen.types.n = 0;
	gen.types.name = NULL;

	if (options->debug->dump_sizes) {
		isl_space *space = isl_space_params_alloc(ctx, 0);
		gen.used_sizes = isl_union_map_empty(space);
	}

	r = ppcg_transform(ctx, input, out, options, &generate_wrap, &gen);

	if (options->debug->dump_sizes) {
		isl_union_map_dump(gen.used_sizes);
		isl_union_map_free(gen.used_sizes);
	}

	isl_union_map_free(gen.sizes);
	for (i = 0; i < gen.types.n; ++i)
		free(gen.types.name[i]);
	free(gen.types.name);

	return r;
}

