#ifndef _POLYSA_PRINT_H
#define _POLYSA_PRINT_H

#include "polysa_common.h"
#include "print.h"

__isl_give isl_printer *polysa_print_types(__isl_take isl_printer *p, 
  struct polysa_types *types, struct polysa_prog *prog);
__isl_give isl_printer *polysa_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node);
__isl_give isl_printer *polysa_array_info_print_declaration_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array,
	const char *memory_space);
__isl_give isl_printer *polysa_array_info_print_call_argument(
	__isl_take isl_printer *p, struct polysa_array_info *array);
__isl_give isl_printer *polysa_kernel_print_domain(__isl_take isl_printer *p,
	struct polysa_kernel_stmt *stmt);
__isl_give isl_printer *polysa_print_local_declarations(__isl_take isl_printer *p,
	struct polysa_prog *prog);
__isl_give isl_printer *polysa_array_info_print_size(__isl_take isl_printer *prn,
	struct polysa_array_info *array);

#endif
