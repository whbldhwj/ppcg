#ifndef _POLYSA_CODEGEN_H
#define _POLYSA_CODEGEN_H

#include "polysa_common.h"
#include "print.h"

__isl_give isl_ast_node *sa_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule);
__isl_give isl_ast_node *sa_module_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule);
__isl_give isl_ast_node *sa_module_call_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule);
__isl_give isl_ast_node *sa_fifo_decl_generate_code(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule);

int polysa_array_requires_device_allocation(struct polysa_array_info *array);
struct polysa_array_tile *polysa_array_ref_group_tile(
	struct polysa_array_ref_group *group);
enum polysa_group_access_type polysa_array_ref_group_type(
	struct polysa_array_ref_group *group);
enum polysa_group_access_type polysa_cpu_array_ref_group_type(
	struct polysa_array_ref_group *group);

#endif
