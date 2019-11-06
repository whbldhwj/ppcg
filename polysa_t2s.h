#ifndef _POLYSA_CPU_H
#define _POLYSA_CPU_H

#include <isl/ctx.h>
#include <isl/id_to_id.h>

#include "ppcg.h"
#include "polysa_common.h"
#include "polysa_vsa.h"
#include "gpu.h"

enum polysa_dep_type {
  POLYSA_DEP_RAW,
  POLYSA_DEP_RAR,
  POLYSA_DEP_WAR,
  POLYSA_DEP_WAW
};

struct polysa_dep {
  isl_id *src; 
  isl_id *dest;
  isl_vec *disvec;
  enum polysa_dep_type type;

  isl_basic_map *isl_dep;

  /* Iteration domain in scheduling dimensions. */
  isl_set *src_sched_domain;
  isl_set *dest_sched_domain;
};

struct t2s_stmt_data {
  int stmt_num;
  struct ppcg_stmt **stmts;
  isl_set **stmt_domain;

  /* The scheduling domain of the statement. */
  isl_set *stmt_anchor_domain;

  /* The mapping from the original iterators to scheduling iterators. */
  isl_pw_multi_aff *iterator_map;

  /* The flow deps associated with each t2s_stmt. */
  struct polysa_dep *** stmt_deps;
  /* The number of access groups. */
  int n_acc_group;
  /* The number of deps in each access group. */
  int *n_dep_per_acc_group;

  /* Internal data for t2s_update_access. 
   * The poitner shares the same content with stmt_deps. 
   * No need for freeup in the destruction of t2s_stmt_data. */
  struct polysa_dep **dep_stmt_pair;
};

struct t2s_URE {
  char *name;
  int update_level;
  char *text;
  int d; // drain URE
};

struct t2s_array_ref_group {
  struct t2s_array_info *t2s_array;
  struct gpu_array_info *array;
  /* Position of this group in the list of reference groups of array. */
  int nr;

  /* The following fields are used during the construction of the groups.
   * write is set if any acccess in the group is a write.
   * exact write is set if all writes are definite writes.
   */
  isl_map *access;
  int write;
  int exact_write;

  /* References in this group; point to elements of a linked list. */
  int n_ref;
  struct gpu_stmt_access **refs;
};

struct t2s_array_info {
  struct gpu_array_info *array;

  int n_group;
  struct t2s_array_ref_group **groups;
};

/* Internal data structure for generating t2s function decls.
 *
 * full_sched is a union map representation of the entire kernel schedule.
 * The schedules are all formulated in terms of the original statement 
 * instances, i.e., those that appear in the domains of the access relations.
 */
struct t2s_group_data {
  isl_union_map *full_sched;
};

struct t2s_data {
  /* The union of scheduling domains of all the statements. */
  isl_set *anchor_domain;  

  /* The scheduling domain of all the statements. */
  isl_union_set *stmt_domain;

  /* The simplified scheduling domain of all the statements. */
  isl_union_set *stmt_sim_domain;

  /* URE. */
  struct t2s_URE **URE;
  int URE_num;

  /* T2S stmt nums. */
  int t2s_stmt_num;  

  /* T2S stmt texts. */
  char **t2s_stmt_text;

  /* Iterator num. */
  int iter_num;

  /* T2S iter. */
  struct polysa_iter **iter;

  /* PPCG scop. */
  struct ppcg_scop *scop;

  /* Printer to print to T2S file. */
  isl_printer *p;

  isl_ctx *ctx;
  
  struct polysa_prog *prog;

  /* Flow deps. */
  struct polysa_dep **deps;
  int ndeps;

  /* Virtual Systolic Array */
  // TODO: struct polysa_vsa *vsa;

  /* Function decls. */
  isl_id_to_id *ref2func;
  // isl_id_to_id *ref2dfunc;
  isl_id_list *func_ids;

  /* Array. */
  int n_array;
  struct t2s_array_info *array;

  /* Used during the construction of statement URE. */
  struct t2s_stmt_data *stmt_data;

  /* Used during the construction of func decls. */
  struct t2s_group_data *group_data;
};

int generate_polysa_t2s(isl_ctx *ctx, struct ppcg_options *ppcg_options,
    const char *input, const char *output);
__isl_give struct polysa_dep *polysa_dep_copy(__isl_keep struct polysa_dep *dep);
void *polysa_dep_free(__isl_take struct polysa_dep *dep);

#endif
