#ifndef _POLYSA_COMMON_H_
#define _POLYSA_COMMON_H_

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "ppcg.h"
//#include "polysa_sa.h"
//#include "polysa_t2s.h"

struct t2s_info {
	FILE *host_c;
	FILE *kernel_c;
  FILE *kernel_h;
};

void t2s_open_files(struct t2s_info *info, const char *input);
void t2s_close_files(struct t2s_info *info);

/* A sequence of "n" names of types.
 */
struct polysa_types {
  int n;
  char **name;
};

struct polysa_prog {
  /* Program schedule */
  isl_schedule *schedule;  

  isl_ctx *ctx;

  struct ppcg_scop *scop;

  /* Array dimension */
  isl_size array_dim;

  /* Band width */
  isl_size array_part_w;
  isl_size space_w;
  isl_size time_w;
};


struct polysa_gen {
	isl_ctx *ctx;
	struct ppcg_options *options;

	/* Callback for printing of AST in appropriate format. */
	__isl_give isl_printer *(*print)(__isl_take isl_printer *p,
		struct polysa_prog *prog, __isl_keep isl_ast_node *tree,
		struct polysa_types *types, void *user);
	void *print_user;

	struct polysa_prog *prog;
	/* The generated AST. */
	isl_ast_node *tree;

	/* The sequence of types for which a definition has been printed. */
	struct polysa_types types;

	/* User specified tile, grid and block sizes for each kernel */
	isl_union_map *sizes;

	/* Effectively used tile, grid and block sizes for each kernel */
	isl_union_map *used_sizes;

	/* Identifier of the next kernel. */
	int kernel_id;
};


struct polysa_vsa {
  int array_part_w;
  int space_w;
  int time_w;

  int t2s_iter_num;
  char **t2s_iters;
};

struct polysa_acc {
  isl_map *tagged_map;
  isl_map *map;
  isl_space *id;

  int rw; // 0 - read 1 - write
};
isl_size isl_union_map_n_basic_map(__isl_keep isl_union_map *umap);
__isl_give isl_basic_map_list *isl_union_map_get_basic_map_list(__isl_keep isl_union_map *umap);
void print_mat(FILE *fp, __isl_keep isl_mat *mat);
isl_bool is_permutable_node_cnt(__isl_keep isl_schedule_node *node, void *user);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
isl_bool sa_legality_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
struct polysa_prog **sa_space_time_transform(__isl_take isl_schedule *schedule, struct ppcg_scop *scop, isl_size *num_sa);
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_permutable_node_update(__isl_keep isl_schedule_node *node, void *user);
isl_stat sa_pe_optimize(struct polysa_prog *sa);
struct polysa_prog *sa_candidates_smart_pick(struct polysa_prog **sa_list, __isl_keep isl_size num_sa);
struct polysa_prog **sa_space_time_transform_at_dim(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_prog **sa_space_time_transform_at_dim_async(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
struct polysa_prog **sa_space_time_transform_at_dim_sync(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop, isl_size dim, isl_size *num_sa);
__isl_give isl_schedule *loop_interchange_at_node(__isl_take isl_schedule_node *node, isl_size level1, isl_size level2);
isl_stat sa_latency_hiding_optimize(struct polysa_prog *sa);
isl_stat sa_SIMD_vectorization_optimize(struct polysa_prog *sa);
isl_stat sa_array_partitioning_optimize(struct polysa_prog *sa);

void *polysa_prog_free(struct polysa_prog *sa);
struct polysa_prog *polysa_prog_copy(struct polysa_prog *sa);
struct polysa_prog *polysa_prog_from_schedule(__isl_take isl_schedule *schedule);

struct polysa_vsa *polysa_vsa_alloc();
void *polysa_vsa_free(struct polysa_vsa *vsa);

void vsa_band_width_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);
void vsa_t2s_iter_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);
void vsa_t2s_var_extract(struct polysa_prog *sa, struct polysa_vsa *vsa);

void *polysa_acc_free(struct polysa_acc *acc);

#endif
