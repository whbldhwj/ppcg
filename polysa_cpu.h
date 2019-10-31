#ifndef _POLYSA_CPU_H
#define _POLYSA_CPU_H

#include <isl/ctx.h>

#include "ppcg.h"
#include "polysa_common.h"
#include "polysa_vsa.h"

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

struct t2s_data {
  /* The union of scheduling domains of all the statements. */
  isl_set *anchor_domain;  

  /* The scheduling domain of all the statements. */
  isl_union_set *stmt_domain;

  /* The simplified scheduling domain of all the statements. */
  isl_union_set *stmt_sim_domain;

  /* T2S stmt nums. */
  int t2s_stmt_num;  

  /* T2S stmt texts. */
  char **t2s_stmt_text;

  /* Iterator num. */
  int iter_num;

  /* PPCG scop. */
  struct ppcg_scop *scop;

  /* Printer to print to T2S file. */
  isl_printer *p;

  isl_ctx *ctx;

  /* Flow deps .*/
  struct polysa_dep **deps;
  int ndeps;

  struct t2s_stmt_data *stmt_data;
};

int generate_polysa_cpu(isl_ctx *ctx, struct ppcg_options *ppcg_options,
    const char *input, const char *output);
__isl_give struct polysa_dep *polysa_dep_copy(__isl_keep struct polysa_dep *dep);
void *polysa_dep_free(__isl_take struct polysa_dep *dep);

#endif
