#ifndef _POLYSA_TRANS_H
#define _POLYSA_TRANS_H

#include <isl/constraint.h>

#include "cpu.h"
#include "polysa_common.h"
#include "print.h"
#include "polysa_codegen.h"
#include "polysa_group.h"

/* Internal data structure for extract_size_of_type.
 * "type" specifies the name of the space that we want to extract.
 * "res" is used to store the subset of that space.
 */
struct polysa_extract_size_data {
	const char *type;
	isl_set *res;
};

/* Internal structure for loop tiling in PE optimization.
 */
struct polysa_pe_opt_tile_data {
  int n_tiled_loop;
  int n_touched_loop;
  int tile_len;
  int *tile_size;
  struct polysa_kernel *sa;
};

/* Array Generation */
struct polysa_kernel **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa);
struct polysa_kernel **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size dim, isl_size *num_sa);    
struct polysa_kernel **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, 
    isl_size dim, isl_size *num_sa);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);

/* PE Optimization */
isl_stat sa_array_partitioning_optimize(struct polysa_kernel *sa);
isl_stat sa_latency_hiding_optimize(struct polysa_kernel *sa);
isl_stat sa_SIMD_vectorization_optimize(struct polysa_kernel *sa);
isl_stat sa_pe_optimize(struct polysa_kernel *sa, bool pass_en[]);
isl_stat sa_loop_init(struct polysa_kernel *sa);
isl_stat sa_space_time_loop_setup(struct polysa_kernel *sa);
struct polysa_kernel **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop,
    isl_size *num_sa);

/* Others */
int generate_sa(isl_ctx *ctx, const char *input, FILE *out, 
  struct ppcg_options *options,
  __isl_give isl_printer *(*print)(__isl_take isl_printer *p,
    struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
    struct polysa_hw_module **modules, int n_modules,
    struct polysa_types *types, void *user), void *user);
__isl_give isl_schedule *sa_map_to_device(struct polysa_gen *gen,
    __isl_take isl_schedule *schedule);
struct polysa_kernel *sa_candidates_smart_pick(struct polysa_kernel **sa_list, __isl_keep isl_size num_sa);

#endif
