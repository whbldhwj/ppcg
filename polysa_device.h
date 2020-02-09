#ifndef _POLYSA_DEVICE_H
#define _POLYSA_DEVICE_H

#include "polysa_common.h"

/* Internal data structure for node_may_persist.
 *
 * "tagger" maps tagged iteration domains to the corresponding untagged
 *	iteration domain.
 *
 * "may_persist_flow" is the set of all tagged dataflow dependences
 * with those dependences removed that either precede or follow
 * the kernel launch in a sequence.
 * "inner_band_flow" is the set of all tagged dataflow dependences
 * that are local to a given iteration of the outer band nodes
 * with respect to the current node.
 * "local_flow" is equal to "inner_band_flow", except that the domain
 * and the range have been intersected with intermediate filters
 * on children of sets or sequences.
 */
struct ppcg_may_persist_data {
	isl_union_pw_multi_aff *tagger;

	isl_union_map *local_flow;
	isl_union_map *inner_band_flow;
	isl_union_map *may_persist_flow;
};

__isl_give isl_schedule_node *sa_add_to_from_device(
  __isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
  __isl_take isl_union_map *prefix, struct polysa_prog *prog);
__isl_give isl_schedule_node *sa_add_init_clear_device(
	__isl_take isl_schedule_node *node);
__isl_give isl_union_map *remove_local_accesses(
	struct polysa_prog *prog, __isl_take isl_union_map *tagged,
	__isl_take isl_union_map *access, __isl_take isl_union_map *sched,
	int read);
__isl_give isl_union_map *remove_local_accesses_flow(
	struct polysa_prog *prog, __isl_take isl_union_map *tagged,
	__isl_take isl_union_map *access, __isl_take isl_union_map *sched,
	int read);
__isl_give isl_union_map *wrapped_reference_to_access(
	__isl_take isl_union_set *ref, __isl_take isl_union_map *tagged);

#endif

