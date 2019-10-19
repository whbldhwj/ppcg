#ifndef _POLYSA_SA_H
#define _POLYSA_SA_H

#include <isl/ast.h>
#include <isl/id.h>
#include <isl/id_to_ast_expr.h>

#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "polysa_common.h"

int generate_sa(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*print)(__isl_take isl_printer *p,
		struct polysa_prog *prog, __isl_keep isl_ast_node *tree,
		struct polysa_types *types, void *user), void *user);

#endif
