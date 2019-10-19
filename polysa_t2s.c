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
#include "polysa_sa.h"

/* This function extracts the iterators for T2S program. */
void vsa_t2s_iter_extract(struct polysa_prog *sa, struct polysa_vsa *vsa) {
  isl_schedule_node *band = get_outermost_permutable_node(sa->schedule);
  isl_size band_w = isl_schedule_node_band_n_member(band);
  
  vsa->t2s_iter_num = band_w;
  vsa->t2s_iters = (char **)malloc(sizeof(char *) * band_w);
  char iter[10];
  for (int i = 0; i < vsa->t2s_iter_num; i++) {
    sprintf(iter, "t%d", i);
    vsa->t2s_iters[i] = strdup(iter);
  }
    
  isl_schedule_node_free(band);
}

static struct polysa_acc **extract_racc_from_tagged_reads(__isl_keep isl_union_map *tagged_reads, int *n_racc) {
  isl_size nreads = isl_union_map_n_basic_map(tagged_reads);
  isl_basic_map_list *racc_list = isl_union_map_get_basic_map_list(tagged_reads);
  struct polysa_acc **raccs = (struct polysa_acc **)malloc(nreads * sizeof(struct polysa_acc *));
  for (int i = 0; i < nreads; i++) {
    raccs[i] = (struct polysa_acc *)malloc(sizeof(struct polysa_acc));
    raccs[i]->tagged_map = isl_map_from_basic_map(isl_basic_map_list_get_basic_map(racc_list, i));
    raccs[i]->map = isl_map_domain_factor_domain(isl_map_copy(raccs[i]->tagged_map));
    isl_space *space = isl_map_get_space(raccs[i]->tagged_map);
    space = isl_space_domain(space);
    space = isl_space_range(space);
    raccs[i]->id = space;
    raccs[i]->rw = 0;
  }
  
  isl_basic_map_list_free(racc_list);
  *n_racc = nreads;

  return raccs;
}

static struct polysa_acc **extract_wacc_from_tagged_writes(__isl_keep isl_union_map *tagged_writes, int *n_wacc) {
  isl_size nwrites = isl_union_map_n_basic_map(tagged_writes);
  isl_basic_map_list *wacc_list = isl_union_map_get_basic_map_list(tagged_writes);
  struct polysa_acc **waccs = (struct polysa_acc **)malloc(nwrites * sizeof(struct polysa_acc *));
  for (int i = 0; i < nwrites; i++) {
    waccs[i] = (struct polysa_acc *)malloc(sizeof(struct polysa_acc));
    waccs[i]->tagged_map = isl_map_from_basic_map(isl_basic_map_list_get_basic_map(wacc_list, i));
    waccs[i]->map = isl_map_domain_factor_domain(isl_map_copy(waccs[i]->tagged_map));
    isl_space *space = isl_map_get_space(waccs[i]->tagged_map);
    space = isl_space_domain(space);
    space = isl_space_range(space);
    waccs[i]->id = space;
    waccs[i]->rw = 0;
  }

  isl_basic_map_list_free(wacc_list);
  *n_wacc = nwrites;

  return waccs;
}

void vsa_t2s_var_extract(struct polysa_prog *sa, struct polysa_vsa *vsa) {
  isl_union_map *tagged_reads = sa->scop->tagged_reads;
  isl_union_map *tagged_may_writes = sa->scop->tagged_may_writes;
  isl_union_map *tagged_must_writes = sa->scop->tagged_must_writes;

  int n_racc, n_wacc;
  struct polysa_acc **raccs = extract_racc_from_tagged_reads(tagged_reads, &n_racc);
  struct polysa_acc **waccs = extract_wacc_from_tagged_writes(tagged_must_writes, &n_wacc);

  for (int i = 0; i < n_racc; i++) {
    polysa_acc_free(raccs[i]);
  }
  for (int i = 0; i < n_wacc; i++) {
    polysa_acc_free(waccs[i]);    
  }
  free(raccs);
  free(waccs);

//  // debug
//  isl_printer *printer = isl_printer_to_file(sa->ctx, stdout);
//  isl_printer_print_union_map(printer, tagged_reads);
//  printf("\n");
//  isl_printer_print_union_map(printer, tagged_may_writes);
//  printf("\n");
//  isl_printer_print_union_map(printer, tagged_must_writes);
//  printf("\n");
//  // debug

//  // initialize the acc_var_map
//  struct acc_var_pair **acc_var_map = NULL;
//  isl_size n_accs = n_racc + n_wacc; 
//
//  acc_var_map = (struct acc_var_pair **)malloc(n_accs * sizeof(struct acc_var_pair *));
//  vsa->acc_var_map = acc_var_map;
//
//  for (int i = 0; i < n_accs; i++) {
//    acc_var_map[i] = (struct acc_var_pair *)malloc(sizeof(struct acc_var_pair));
//    acc_var_map[i]->ei = -1;
//    acc_var_map[i]->d = 0;
//    acc_var_map[i]->var_name = NULL;
//    acc_var_map[i]->var_ref = NULL;
//    acc_var_map[i]->var_iters = (IterExp **)malloc(vsa->t2s_iter_num * sizeof(IterExp *));
//    for (int iter_id = 0; iter_id < vsa->t2s_iter_num; iter_id++) {
//      acc_var_map[i]->var_iters[iter_id] = (IterExp *)malloc(sizeof(IterExp));
//      acc_var_map[i]->var_iters[iter_id]->iter_name = strdup(vsa->t2s_iters[iter_id]);
//      acc_var_map[i]->var_iters[iter_id]->iter_offset = 0;
//    }    
//  }

   
}

static __isl_give isl_printer *print_t2s(__isl_take isl_printer *p,
		struct polysa_prog *prog, __isl_keep isl_ast_node *tree,
		struct polysa_types *types, void *user) {
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
