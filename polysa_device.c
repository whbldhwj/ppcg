#include "polysa_device.h"

///* Construct the string "<a>_<b>".
// */
//static char *concat(isl_ctx *ctx, const char *a, const char *b)
//{
//	isl_printer *p;
//	char *s;
//
//	p = isl_printer_to_str(ctx);
//	p = isl_printer_print_str(p, a);
//	p = isl_printer_print_str(p, "_");
//	p = isl_printer_print_str(p, b);
//	s = isl_printer_get_str(p);
//	isl_printer_free(p);
//
//	return s;
//}

/* Given a set of wrapped references "ref", return the corresponding
 * access relations based on the tagged access relations "tagged".
 *
 * The elements of "ref" are of the form
 *
 *	[D -> R]
 *
 * with D an iteration domains and R a reference.
 * The elements of "tagged" are of the form
 *
 *	[D -> R] -> A
 *
 * with A an array.
 *
 * Extend "tagged" to include the iteration domain in the range, i.e.,
 *
 *	[D -> R] -> [D -> A]
 *
 * apply the result to "ref" and then unwrap the resulting set
 * to obtain relations of the form
 *
 *	D -> A
 */
__isl_give isl_union_map *wrapped_reference_to_access(
	__isl_take isl_union_set *ref, __isl_take isl_union_map *tagged)
{
	isl_union_map *tag2access;

	tag2access = isl_union_map_copy(tagged);
	tag2access = isl_union_map_universe(tag2access);
	tag2access = isl_union_set_unwrap(isl_union_map_domain(tag2access));
//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_map_get_ctx(tag2access), stdout);
//  p = isl_printer_print_union_map(p, tag2access);
//  printf("\n");
//  // debug
  /* Construct [D -> R] -> D */
	tag2access = isl_union_map_domain_map(tag2access);
//  // debug
//  p = isl_printer_print_union_map(p, tag2access);
//  printf("\n"); 
//  p = isl_printer_print_union_map(p, tagged);
//  printf("\n"); 
//  // debug
  /* Construct [D -> R] -> [D -> A] */
	tag2access = isl_union_map_range_product(tag2access, tagged);
//  // debug
//  p = isl_printer_print_union_map(p, tag2access);
//  printf("\n"); 
//  // debug

	ref = isl_union_set_coalesce(ref);
	ref = isl_union_set_apply(ref, tag2access);

	return isl_union_set_unwrap(ref);
}

/* Given an access relation "access" from one or more array reference groups,
 * remove those reads if ("read" is 1) or writes (if "read" is 0)
 * that are only needed to communicate data within
 * the same iteration of "sched".
 * The domain of "sched" corresponds to the original statement instances,
 * i.e., those that appear in the domains of the access relations.
 * "tagged" contains all tagged access relations to all
 * the array reference groups accessed by "access" from statement
 * instances scheduled by "sched".
 *
 * If the access is a read then it is either an element of
 *
 *	live_in union (range flow)
 *
 * where live_in and flow may be overapproximations, or
 * it reads an uninitialized value (that is not live-in because
 * there is an intermediate kill) or it reads a value that was
 * written within the same (compound) statement instance.
 * If the access is a write then it is either an element of
 *
 *	live_out union (domain flow)
 *
 * or it writes a value that is never read (and is not live-out
 * because of an intermediate kill) or only
 * within the same (compound) statement instance.
 * In both cases, the access relation is also a subset of
 * the group access relation.
 *
 * The cases where an uninitialized value is read or a value is written
 * that is never read or where the dataflow occurs within a statement
 * instance are also considered local and may also be removed.
 *
 * Essentially, we compute the intersection of "access" with either
 *
 *	live_in union (range non-local-flow)
 *
 * or
 *
 *	live_out union (domain non-local-flow)
 *
 * We first construct a relation "local"
 *
 *	[[D -> R] -> [D' -> R']]
 *
 * of pairs of domain iterations accessing the reference group
 * and references in the group that are coscheduled by "sched".
 *
 * If this relation does not intersect the dataflow dependences,
 * then there is nothing we can possibly remove, unless the dataflow
 * dependences themselves only relate a subset of the accesses.
 * In particular, the accesses may not be involved in any dataflow
 * dependences, either because they are uninitialized reads/dead writes
 * or because the dataflow occurs inside a statement instance.
 *
 * Since the computation below may break up the access relation
 * into smaller pieces, we only perform the intersection with
 * the non-local dependent accesses if the local pairs
 * intersect the dataflow dependences. Otherwise, we intersect
 * with the universe of the non-local dependent accesses.
 * This should at least remove accesses from statements that
 * do not participate in any dependences.
 *
 * In particular, we remove the "local" dataflow dependences from
 * the set of all dataflow dependences, or at least those
 * that may contribute to a domain/range that intersects
 * the domain of "access".
 * Note that if the potential dataflow dependences are an overapproximation
 * of the actual dataflow dependences, then the result remains an
 * overapproximation of the non-local dataflow dependences.
 * Copying to/from global memory is only needed for the references
 * in the domain/range of the result or for accesses that are live out/in
 * for the entire scop.
 *
 * We therefore map the domain/range of the "external" relation
 * to the corresponding access relation and take the union with
 * the live out/in relation.
 */
__isl_give isl_union_map *remove_local_accesses(
	struct polysa_prog *prog, __isl_take isl_union_map *tagged,
	__isl_take isl_union_map *access, __isl_take isl_union_map *sched,
	int read)
{
	int empty;
	isl_union_pw_multi_aff *tagger;
	isl_union_set *domain, *access_domain;
	isl_union_map *local, *external, *universe;
	isl_union_set *tag_set;

	if (isl_union_map_is_empty(access)) {
		isl_union_map_free(sched);
		isl_union_map_free(tagged);
		return access;
	}

  /* Tagger maps the tagged iteration domain to untagged iteration domain. 
   * Iteration domain is tagged to the access function.
   * e.g., [S1[i,j,k]->_pet_ref_1[]] -> S1[(i),(j),(k)]
   */
	tagger = isl_union_pw_multi_aff_copy(prog->scop->tagger);
	domain = isl_union_map_domain(isl_union_map_copy(tagged));
	tagger = isl_union_pw_multi_aff_intersect_domain(tagger,
					isl_union_set_copy(domain));
	sched = isl_union_map_preimage_domain_union_pw_multi_aff(sched, tagger);

  /* Construct the relation "local"
   * [[D -> R] -> [D' -> R']]
   */
	local = isl_union_map_apply_range(sched,
			    isl_union_map_reverse(isl_union_map_copy(sched)));
  /* Derive the local dependence set. */
	local = isl_union_map_intersect(local,
			isl_union_map_copy(prog->scop->tagged_dep_flow));

	empty = isl_union_map_is_empty(local);

	external = isl_union_map_copy(prog->scop->tagged_dep_flow);
	universe = isl_union_map_universe(isl_union_map_copy(access));
	access_domain = isl_union_map_domain(universe);
	domain = isl_union_set_universe(domain);
	universe = isl_union_set_unwrap(domain);
	universe = isl_union_map_intersect_domain(universe, access_domain);
	domain = isl_union_map_wrap(universe);
	if (read)
		external = isl_union_map_intersect_range(external, domain);
	else
		external = isl_union_map_intersect_domain(external, domain);
	external = isl_union_map_intersect_params(external,
				isl_set_copy(prog->scop->context));
	external = isl_union_map_subtract(external, local);
  /* So far external contains only access non-local RAW pairs. */

	if (read) {
		tag_set = isl_union_map_range(external);
		external = wrapped_reference_to_access(tag_set, tagged);
    /* Temporarily commented out, we don't consider live-in so far. */
		external = isl_union_map_union(external,
				isl_union_map_copy(prog->scop->live_in));
	} else {
		tag_set = isl_union_map_domain(external);
		external = wrapped_reference_to_access(tag_set, tagged);
    /* Temporarily commented out, we don't consider live-out so far. */
		external = isl_union_map_union(external,
				isl_union_map_copy(prog->scop->live_out));
	}

	if (empty < 0)
		external = isl_union_map_free(external);
	else if (empty)
		external = isl_union_map_universe(external);

	access = isl_union_map_intersect(access, external);

	return access;
}

/* Extended from remove_local_accesses.
 * Excluding live-in and live-out, this function only considers
 * RAW deps.
 */
__isl_give isl_union_map *remove_local_accesses_flow(
	struct polysa_prog *prog, __isl_take isl_union_map *tagged,
	__isl_take isl_union_map *access, __isl_take isl_union_map *sched,
	int read)
{
	int empty;
	isl_union_pw_multi_aff *tagger;
	isl_union_set *domain, *access_domain;
	isl_union_map *local, *external, *universe;
	isl_union_set *tag_set;

	if (isl_union_map_is_empty(access)) {
		isl_union_map_free(sched);
		isl_union_map_free(tagged);
		return access;
	}

  /* Tagger maps the tagged iteration domain to untagged iteration domain. 
   * Iteration domain is tagged to the access function.
   * e.g., [S1[i,j,k]->_pet_ref_1[]] -> S1[(i),(j),(k)]
   */
	tagger = isl_union_pw_multi_aff_copy(prog->scop->tagger);
	domain = isl_union_map_domain(isl_union_map_copy(tagged));
	tagger = isl_union_pw_multi_aff_intersect_domain(tagger,
					isl_union_set_copy(domain));
	sched = isl_union_map_preimage_domain_union_pw_multi_aff(sched, tagger);

  /* Construct the relation "local"
   * [[D -> R] -> [D' -> R']]
   */
	local = isl_union_map_apply_range(sched,
			    isl_union_map_reverse(isl_union_map_copy(sched)));

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_union_map_get_ctx(external), stdout);
//  p = isl_printer_print_union_map(p, local);
//  printf("\n");
//  // debug

  /* Derive the local dependence set. */
	local = isl_union_map_intersect(local,
			isl_union_map_copy(prog->scop->tagged_dep_flow));

//  // debug
//  p = isl_printer_print_union_map(p, local);
//  printf("\n");
//  // debug

	empty = isl_union_map_is_empty(local);

	external = isl_union_map_copy(prog->scop->tagged_dep_flow);
	universe = isl_union_map_universe(isl_union_map_copy(access));
	access_domain = isl_union_map_domain(universe);
	domain = isl_union_set_universe(domain);
	universe = isl_union_set_unwrap(domain);
	universe = isl_union_map_intersect_domain(universe, access_domain);
	domain = isl_union_map_wrap(universe);
	if (read)
		external = isl_union_map_intersect_range(external, domain);
	else
		external = isl_union_map_intersect_domain(external, domain);
	external = isl_union_map_intersect_params(external,
				isl_set_copy(prog->scop->context));
//  // debug
//  p = isl_printer_print_union_map(p, external);
//  printf("\n");
//  // debug
	external = isl_union_map_subtract(external, local);
  /* So far external contains only access non-local RAW pairs. */

//  // debug
////  isl_printer *p = isl_printer_to_file(isl_union_map_get_ctx(external), stdout);
//  p = isl_printer_print_union_map(p, external);
//  printf("\n");
//  p = isl_printer_print_union_map(p, prog->scop->tagged_dep_flow);
//  printf("\n");
//  p = isl_printer_free(p);
//  // debug

	if (read) {
		tag_set = isl_union_map_range(external);
		external = wrapped_reference_to_access(tag_set, tagged);
//    /* Temporarily commented out, we don't consider live-in so far. */
//		external = isl_union_map_union(external,
//				isl_union_map_copy(prog->scop->live_in));
	} else {
		tag_set = isl_union_map_domain(external);
		external = wrapped_reference_to_access(tag_set, tagged);
//    /* Temporarily commented out, we don't consider live-out so far. */
//		external = isl_union_map_union(external,
//				isl_union_map_copy(prog->scop->live_out));
	}

	if (empty < 0)
		external = isl_union_map_free(external);
	else if (empty)
		external = isl_union_map_universe(external);

	access = isl_union_map_intersect(access, external);

	return access;
}


/* Replace any reference to an array element in the range of "copy"
 * by a reference to all array elements (defined by the extent of the array).
 */
static __isl_give isl_union_map *approximate_copy_out(
	__isl_take isl_union_map *copy, struct polysa_prog *prog)
{
	int i;
	isl_union_map *res;

	res = isl_union_map_empty(isl_union_map_get_space(copy));

	for (i = 0; i < prog->n_array; ++i) {
		isl_space *space;
		isl_set *set;
		isl_union_map *copy_i;
		isl_union_set *extent, *domain;

		space = isl_space_copy(prog->array[i].space);
		extent = isl_union_set_from_set(isl_set_universe(space));
		copy_i = isl_union_map_copy(copy);
		copy_i = isl_union_map_intersect_range(copy_i, extent);
		set = isl_set_copy(prog->array[i].extent);
		extent = isl_union_set_from_set(set);
		domain = isl_union_map_domain(copy_i);
		copy_i = isl_union_map_from_domain_and_range(domain, extent);
		res = isl_union_map_union(res, copy_i);
	}

	isl_union_map_free(copy);

	return res;
}

/* Return (the universe spaces of) the arrays that are declared
 * inside the scop corresponding to "prog" and for which all
 * potential writes inside the scop form a subset of "domain".
 */
static __isl_give isl_union_set *extract_local_accesses(struct polysa_prog *prog,
	__isl_keep isl_union_set *domain)
{
	int i;
	isl_union_set *local;

	local = isl_union_set_empty(isl_union_set_get_space(domain));

	for (i = 0; i < prog->n_array; ++i) {
		isl_set *set;
		isl_union_map *to_outer;
		isl_union_map *may_write;
		isl_union_set *write_domain;
		isl_union_set *fields;
		int subset;

		if (!prog->array[i].local)
			continue;

		set = isl_set_universe(isl_space_copy(prog->array[i].space));
		to_outer = isl_union_map_copy(prog->to_outer);
		to_outer = isl_union_map_intersect_range(to_outer,
				    isl_union_set_from_set(isl_set_copy(set)));
		fields = isl_union_map_domain(to_outer);
		may_write = isl_union_map_copy(prog->may_write);
		may_write = isl_union_map_intersect_range(may_write, fields);
		write_domain = isl_union_map_domain(may_write);
		subset = isl_union_set_is_subset(write_domain, domain);
		isl_union_set_free(write_domain);

		if (subset < 0) {
			isl_set_free(set);
			return isl_union_set_free(local);
		} else if (subset) {
			local = isl_union_set_add_set(local, set);
		} else {
			isl_set_free(set);
		}
	}

	return local;
}

/* Update the information in "data" based on the band ancestor "node".
 *
 * In particular, we restrict the dependences in data->local_flow
 * to those dependence where the source and the sink occur in
 * the same iteration of the given band node.
 * We also update data->inner_band_flow to the new value of
 * data->local_flow.
 */
static int update_may_persist_at_band(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data)
{
	isl_multi_union_pw_aff *partial;
	isl_union_pw_multi_aff *contraction;
	isl_union_map *flow;

	if (isl_schedule_node_band_n_member(node) == 0)
		return 0;

	partial = isl_schedule_node_band_get_partial_schedule(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
								contraction);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
				isl_union_pw_multi_aff_copy(data->tagger));

	flow = data->local_flow;
	flow = isl_union_map_eq_at_multi_union_pw_aff(flow, partial);
	data->local_flow = flow;

	isl_union_map_free(data->inner_band_flow);
	data->inner_band_flow = isl_union_map_copy(data->local_flow);

	return 0;
}

/* Given a set of local reaching domain elements "domain",
 * expand them to the corresponding leaf domain elements using "contraction"
 * and insert the array references tags using data->tagger.
 */
static __isl_give isl_union_set *expand_and_tag(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_pw_multi_aff *contraction,
	struct ppcg_may_persist_data *data)
{
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
			    contraction);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
			    isl_union_pw_multi_aff_copy(data->tagger));
	return domain;
}

/* Given a filter node that is the child of a set or sequence node,
 * restrict data->local_flow to refer only to those elements
 * in the filter of the node.
 * "contraction" maps the leaf domain elements of the schedule tree
 * to the corresponding domain elements at (the parent of) "node".
 */
static int filter_flow(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data,
	__isl_take isl_union_pw_multi_aff *contraction)
{
	isl_union_set *filter;
	isl_union_map *flow;

	flow = data->local_flow;
	filter = isl_schedule_node_filter_get_filter(node);
	filter = expand_and_tag(filter, contraction, data);
	flow = isl_union_map_intersect_domain(flow, isl_union_set_copy(filter));
	flow = isl_union_map_intersect_range(flow, filter);
	data->local_flow = flow;

	return 0;
}

/* Given a filter node "node", collect the filters on all preceding siblings
 * (which are also filter nodes), add them to "filters" and return the result.
 */
static __isl_give isl_union_set *add_previous_filters(
	__isl_take isl_union_set *filters, __isl_keep isl_schedule_node *node)
{
	isl_schedule_node *sibling;

	sibling = isl_schedule_node_copy(node);
	while (sibling && isl_schedule_node_has_previous_sibling(sibling)) {
		isl_union_set *filter;

		sibling = isl_schedule_node_previous_sibling(sibling);
		filter = isl_schedule_node_filter_get_filter(sibling);
		filters = isl_union_set_union(filters, filter);
	}
	isl_schedule_node_free(sibling);
	if (!sibling)
		return isl_union_set_free(filters);

	return filters;
}

/* Given a filter node "node", collect the filters on all following siblings
 * (which are also filter nodes), add them to "filters" and return the result.
 */
static __isl_give isl_union_set *add_next_filters(
	__isl_take isl_union_set *filters, __isl_keep isl_schedule_node *node)
{
	isl_schedule_node *sibling;

	sibling = isl_schedule_node_copy(node);
	while (sibling && isl_schedule_node_has_next_sibling(sibling)) {
		isl_union_set *filter;

		sibling = isl_schedule_node_next_sibling(sibling);
		filter = isl_schedule_node_filter_get_filter(sibling);
		filters = isl_union_set_union(filters, filter);
	}
	isl_schedule_node_free(sibling);
	if (!sibling)
		return isl_union_set_free(filters);

	return filters;
}

/* Remove those flow dependences from data->may_persist_flow
 * that flow between elements of "domain" within the same iteration
 * of all outer band nodes.
 * "contraction" maps the leaf domain elements of the schedule tree
 * to the corresponding elements "domain".
 */
static void remove_external_flow(struct ppcg_may_persist_data *data,
	__isl_take isl_union_set *domain,
	__isl_keep isl_union_pw_multi_aff *contraction)
{
	isl_union_map *flow;

	contraction = isl_union_pw_multi_aff_copy(contraction);
	domain = expand_and_tag(domain, contraction, data);
	flow = isl_union_map_copy(data->local_flow);
	flow = isl_union_map_intersect_domain(flow, isl_union_set_copy(domain));
	flow = isl_union_map_intersect_range(flow, domain);

	data->may_persist_flow = isl_union_map_subtract(data->may_persist_flow,
							flow);
}

/* Update the information in "data" based on the filter ancestor "node".
 * We only need to modify anything if the filter is the child
 * of a set or sequence node.
 *
 * In the case of a sequence, we remove the dependences between
 * statement instances that are both executed either before or
 * after the subtree that will be mapped to a kernel, within
 * the same iteration of outer bands.
 *
 * In both cases, we restrict data->local_flow to the current child.
 */
static int update_may_persist_at_filter(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data)
{
	enum isl_schedule_node_type type;
	isl_schedule_node *parent;
	isl_space *space;
	isl_union_pw_multi_aff *contraction;
	isl_union_set *before, *after, *filter;

	type = isl_schedule_node_get_parent_type(node);
	if (type != isl_schedule_node_sequence && type != isl_schedule_node_set)
		return 0;

	parent = isl_schedule_node_copy(node);
	parent = isl_schedule_node_parent(parent);
	contraction = isl_schedule_node_get_subtree_contraction(parent);
	isl_schedule_node_free(parent);

	if (type == isl_schedule_node_set)
		return filter_flow(node, data, contraction);

	filter = isl_schedule_node_filter_get_filter(node);
	space = isl_union_set_get_space(filter);
	isl_union_set_free(filter);
	before = isl_union_set_empty(space);
	after = isl_union_set_copy(before);
	before = add_previous_filters(before, node);
	after = add_next_filters(after, node);

	remove_external_flow(data, before, contraction);
	remove_external_flow(data, after, contraction);

	return filter_flow(node, data, contraction);
}

/* Update the information in "data" based on the ancestor "node".
 */
static isl_stat update_may_persist_at(__isl_keep isl_schedule_node *node,
	void *user)
{
	struct ppcg_may_persist_data *data = user;

	switch (isl_schedule_node_get_type(node)) {
	case isl_schedule_node_error:
		return isl_stat_error;
	case isl_schedule_node_context:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_guard:
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	case isl_schedule_node_band:
		if (update_may_persist_at_band(node, data) < 0)
			return isl_stat_error;
		break;
	case isl_schedule_node_filter:
		if (update_may_persist_at_filter(node, data) < 0)
			return isl_stat_error;
		break;
	}

	return isl_stat_ok;
}

/* Determine the set of array elements that may need to be perserved
 * by a kernel constructed from the subtree at "node".
 * This includes the set of array elements that may need to be preserved
 * by the entire scop (prog->may_persist) and the elements for which
 * there is a potential flow dependence that may cross a kernel launch.
 *
 * To determine the second set, we start from all flow dependences.
 * From this set of dependences, we remove those that cannot possibly
 * require data to be preserved by a kernel launch.
 * In particular, we consider the following sets of dependences.
 * - dependences of which the write occurs inside the kernel.
 *   If the data is needed outside the kernel, then it will
 *   be copied out immediately after the kernel launch, so there
 *   is no need for any special care.
 * - dependences of which the read occurs inside the kernel and the
 *   corresponding write occurs inside the same iteration of the
 *   outer band nodes.  This means that the data is needed in
 *   the first kernel launch after the write, which is already
 *   taken care of by the standard copy-in.  That is, the data
 *   do not need to be preserved by any intermediate call to
 *   the same kernel.
 * - dependences of which the write and the read either both occur
 *   before the kernel launch or both occur after the kernel launch,
 *   within the same iteration of the outer band nodes with respect
 *   to the sequence that determines the ordering of the dependence
 *   and the kernel launch.  Such flow dependences cannot cross
 *   any kernel launch.
 *
 * For the remaining (tagged) dependences, we take the domain
 * (i.e., the tagged writes) and apply the tagged access relation
 * to obtain the accessed data elements.
 * These are then combined with the elements that may need to be
 * preserved by the entire scop.
 */
static __isl_give isl_union_set *node_may_persist(
	__isl_keep isl_schedule_node *node, struct polysa_prog *prog)
{
	struct ppcg_may_persist_data data;
	isl_union_pw_multi_aff *contraction;
	isl_union_set *domain;
	isl_union_set *persist;
	isl_union_map *flow, *local_flow;

	data.tagger = prog->scop->tagger;

	flow = isl_union_map_copy(prog->scop->tagged_dep_flow);
	data.local_flow = isl_union_map_copy(flow);
	data.inner_band_flow = isl_union_map_copy(flow);
	data.may_persist_flow = flow;
	if (isl_schedule_node_foreach_ancestor_top_down(node,
					&update_may_persist_at, &data) < 0) 
		data.may_persist_flow =
				    isl_union_map_free(data.may_persist_flow);
	flow = data.may_persist_flow;
	isl_union_map_free(data.local_flow);

	domain = isl_schedule_node_get_domain(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
				    contraction);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
				    isl_union_pw_multi_aff_copy(data.tagger));
  /* Substract the case 1. */ 
	flow = isl_union_map_subtract_domain(flow, isl_union_set_copy(domain)); 
	local_flow = data.inner_band_flow;
	local_flow = isl_union_map_intersect_range(local_flow, domain);
  /* Substract the case 2. */
	flow = isl_union_map_subtract(flow, local_flow);

	persist = isl_union_map_domain(flow);
	persist = isl_union_set_apply(persist,
			isl_union_map_copy(prog->scop->tagged_may_writes));
	persist = isl_union_set_union(persist,
			isl_union_set_copy(prog->may_persist));

	return persist;
}

/* For each array in "prog" of which an element appears in "accessed" and
 * that is not a read only scalar, create a zero-dimensional universe set
 * of which the tuple id has name "<prefix>_<name of array>" and a user
 * pointer pointing to the array (polysa_array_info).
 *
 * If the array is local to "prog", then make sure it will be declared
 * in the host code.
 *
 * Return the list of these universe sets.
 */
static __isl_give isl_union_set_list *create_copy_filters(struct polysa_prog *prog,
	const char *prefix, __isl_take isl_union_set *accessed)
{
	int i;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	ctx = prog->ctx;
	filters = isl_union_set_list_alloc(ctx, 0);
	for (i = 0; i < prog->n_array; ++i) {
		struct polysa_array_info *array = &prog->array[i];
		isl_space *space;
		isl_set *accessed_i;
		int empty;
		char *name;
		isl_id *id;
		isl_union_set *uset;

		if (polysa_array_is_read_only_scalar(array))
			continue;

		space = isl_space_copy(array->space);
		accessed_i = isl_union_set_extract_set(accessed, space);
		empty = isl_set_plain_is_empty(accessed_i);
		isl_set_free(accessed_i);
		if (empty < 0) {
			filters = isl_union_set_list_free(filters);
			break;
		}
		if (empty)
			continue;

		array->global = 1;
		if (array->local)
			array->declare_local = 1;

		name = concat(ctx, prefix, array->name);
		id = name ? isl_id_alloc(ctx, name, array) : NULL;
		free(name);
		space = isl_space_set_alloc(ctx, 0, 0);
		space = isl_space_set_tuple_id(space, isl_dim_set, id);
		uset = isl_union_set_from_set(isl_set_universe(space));

		filters = isl_union_set_list_add(filters, uset);
	}
	isl_union_set_free(accessed);

	return filters;
}

/* Return the set of parameter values for which the array has a positive
 * size in all dimensions.
 * If the sizes are only valid for some parameter values, then those
 * constraints are also taken into account.
 */
__isl_give isl_set *polysa_array_positive_size_guard(struct polysa_array_info *array)
{
	int i;
	isl_space *space;
	isl_set *guard;

	if (!array)
		return NULL;

	space = isl_space_params(isl_space_copy(array->space));
	guard = isl_set_universe(space);

	for (i = 0; i < array->n_index; ++i) {
		isl_pw_aff *bound;
		isl_set *guard_i, *zero;

		bound = isl_multi_pw_aff_get_pw_aff(array->bound, i);
		guard_i = isl_pw_aff_nonneg_set(isl_pw_aff_copy(bound));
		zero = isl_pw_aff_zero_set(bound);
		guard_i = isl_set_subtract(guard_i, zero);
		guard = isl_set_intersect(guard, guard_i);
	}

	return guard;
}

/* Make sure that code for the statements in "filters" that
 * copy arrays to or from the device is only generated when
 * the size of the corresponding array is positive.
 * That is, add a set node underneath "graft" with "filters" as children
 * and for each child add a guard that the selects the parameter
 * values for which the corresponding array has a positive size.
 * The array is available in the user pointer of the statement identifier.
 * "depth" is the schedule depth of the position where "graft"
 * will be added.
 */
static __isl_give isl_schedule_node *insert_positive_size_guards(
	__isl_take isl_schedule_node *graft,
	__isl_take isl_union_set_list *filters, int depth)
{
	int i, n;

	graft = isl_schedule_node_child(graft, 0);
	graft = isl_schedule_node_insert_set(graft, filters);
	n = isl_schedule_node_n_children(graft);
	for (i = 0; i < n; ++i) {
		isl_union_set *filter;
		isl_set *domain, *guard;
		isl_id *id;
		struct polysa_array_info *array;

		graft = isl_schedule_node_child(graft, i);
		filter = isl_schedule_node_filter_get_filter(graft);
		domain = isl_set_from_union_set(filter);
		id = isl_set_get_tuple_id(domain);
		array = isl_id_get_user(id);
		isl_id_free(id);
		isl_set_free(domain);
		guard = polysa_array_positive_size_guard(array);
		guard = isl_set_from_params(guard);
		guard = isl_set_add_dims(guard, isl_dim_set, depth);
		graft = isl_schedule_node_child(graft, 0);
		graft = isl_schedule_node_insert_guard(graft, guard);
		graft = isl_schedule_node_parent(graft);
		graft = isl_schedule_node_parent(graft);
	}
	graft = isl_schedule_node_parent(graft);

	return graft;
}

/* Create a graft for copying arrays to or from the device,
 * whenever the size of the array is strictly positive.
 * Each statement is called "<prefix>_<name of array>" and
 * the identifier has a user pointer pointing to the array.
 * The graft will be added at the position specified by "node".
 * "copy" contains the array elements that need to be copied.
 * Only arrays of which some elements need to be copied
 * will have a corresponding statement in the graph.
 * Note though that each such statement will copy the entire array.
 */
static __isl_give isl_schedule_node *create_copy_device(struct polysa_prog *prog,
	__isl_keep isl_schedule_node *node, const char *prefix,
	__isl_take isl_union_set *copy)
{
	int depth;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *all, *domain;
	isl_union_set_list *filters;
	isl_union_map *extension;
	isl_schedule_node *graft;

	ctx = prog->ctx;
	depth = isl_schedule_node_get_schedule_depth(node);
	filters = create_copy_filters(prog, prefix, copy);
	all = isl_union_set_list_union(isl_union_set_list_copy(filters));

	space = depth < 0 ? NULL : isl_space_set_alloc(ctx, 0, depth);
	domain = isl_union_set_from_set(isl_set_universe(space));
	extension = isl_union_map_from_domain_and_range(domain, all);
	graft = isl_schedule_node_from_extension(extension);

	if (!filters)
		return isl_schedule_node_free(graft);
	if (isl_union_set_list_n_union_set(filters) == 0) {
		isl_union_set_list_free(filters);
		return graft;
	}

	return insert_positive_size_guards(graft, filters, depth);
}


/* Add nodes for copying outer arrays in and out of the device
 * before and after the subtree "node", which contains one or more kernels.
 * "domain" contains the original statement instances, i.e.,
 * those that correspond to the domains of the access relations in "prog".
 * In particular, the domain has not been contracted in any way.
 * "prefix" contains the prefix schedule at that point, in terms
 * of the same original statement instances.
 *
 * We first compute the sets of outer array elements that need
 * to be copied in and out and then graft in the nodes for
 * performing this copying.
 *
 * In particular, for each array that is possibly written anywhere in
 * the subtree "node" and that may be used after "node"
 * or that may be visible outside the corresponding scop,
 * we copy out its entire extent.
 *
 * Any array elements that is read without first being written inside
 * the subtree "node" needs to be copied in.
 * Furthermore, if there are any array elements that
 * are copied out, but that may not be written inside "node", then
 * they also need to be copied in to ensure that the value after execution
 * is the same as the value before execution, at least for those array
 * elements that may have their values preserved by the scop or that
 * may be written before "node" and read after "node".
 * In case the array elements are structures, we need to take into
 * account that all members of the structures need to be written
 * by "node" before we can avoid copying the data structure in.
 *
 * Note that the may_write relation is intersected with the domain,
 * which has been intersected with the context.
 * This helps in those cases where the arrays are declared with a fixed size,
 * while the accesses are parametric and the context assigns a fixed value
 * to the parameters.
 *
 * If an element from a local array is read without first being written,
 * then there is no point in copying it in since it cannot have been
 * written prior to the scop. Warn about the uninitialized read instead.
 */
__isl_give isl_schedule_node *sa_add_to_from_device(
  __isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
  __isl_take isl_union_map *prefix, struct polysa_prog *prog)
{
  isl_union_set *local;
  isl_union_set *may_persist;
  isl_union_map *may_write, *must_write, *copy_out, *not_written;
  isl_union_map *read, *copy_in;
  isl_union_map *tagged;
  isl_union_map *local_uninitialized;
  isl_schedule_node *graft;

  /* Compute the copy-out that contains the live-out union
   * domain of non-local flow dep. 
   */
  tagged = isl_union_map_copy(prog->scop->tagged_reads);
  tagged = isl_union_map_union(tagged,
            isl_union_map_copy(prog->scop->tagged_may_writes));
  may_write = isl_union_map_copy(prog->may_write);
  may_write = isl_union_map_intersect_domain(may_write,
      isl_union_set_copy(domain));
  /* Keep ouly the live-out union domain of non-local flow. */
  may_write = remove_local_accesses(prog,
      isl_union_map_copy(tagged), may_write,
      isl_union_map_copy(prefix), 0);
  may_write = isl_union_map_apply_range(may_write,
      isl_union_map_copy(prog->to_outer));
  may_write = isl_union_map_apply_domain(may_write,
      isl_union_map_copy(prefix));
  may_write = approximate_copy_out(may_write, prog); 
  copy_out = isl_union_map_copy(may_write);

  /* Compute the copy-in. */
  may_write = isl_union_map_apply_range(may_write,
      isl_union_map_copy(prog->to_inner));
  must_write = isl_union_map_copy(prog->must_write);
  must_write = isl_union_map_apply_domain(must_write,
      isl_union_map_copy(prefix));

  may_persist = node_may_persist(node, prog); 
  may_write = isl_union_map_intersect_range(may_write, may_persist);
  not_written = isl_union_map_subtract(may_write, must_write);

  /* Detect the unitialized reads. */
  /* "local" contains (universal space) of arrays that are declared locally and 
   * written by "domain". */
  local = extract_local_accesses(prog, domain); 
  local = isl_union_set_apply(local, isl_union_map_copy(prog->to_inner));
  local_uninitialized = isl_union_map_copy(prog->scop->live_in);
  /* The local unitialized is defined as a read of a local array without 
   * first being written. */
  local_uninitialized = isl_union_map_intersect_range(local_uninitialized,
      local);
  read = isl_union_map_copy(prog->read);
  read = isl_union_map_intersect_domain(read, domain);
  read = remove_local_accesses(prog, tagged, read,
      isl_union_map_copy(prefix), 1);
  local_uninitialized = isl_union_map_intersect(local_uninitialized,
      isl_union_map_copy(read));
  if (!isl_union_map_is_empty(local_uninitialized)) {
    fprintf(stderr,
        "possibly uninitialized reads (not copied in):\n");
    isl_union_map_dump(local_uninitialized);
  }
  read = isl_union_map_subtract(read, local_uninitialized);
  read = isl_union_map_apply_domain(read, prefix);
  copy_in = isl_union_map_union(read, not_written);
  copy_in = isl_union_map_apply_range(copy_in,
      isl_union_map_copy(prog->to_outer));

//  // debug
//  isl_printer *p = isl_printer_to_file(isl_schedule_node_get_ctx(node), stdout);
//  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  /* Add in the copy-in/copy-out nodes. */
  graft = create_copy_device(prog, node, "to_device",
      isl_union_map_range(copy_in)); 
  node = isl_schedule_node_graft_before(node, graft);
  graft = create_copy_device(prog, node, "from_device",
      isl_union_map_range(copy_out)); 
  node = isl_schedule_node_graft_after(node, graft);
 
//  // debug
//  p = isl_printer_print_schedule_node(p, node);
//  printf("\n");
//  // debug

  return node;
}

/* Add nodes for initializing ("init_device") and clearing ("clear_device")
 * the device before and after "node".
 */
__isl_give isl_schedule_node *sa_add_init_clear_device(
	__isl_take isl_schedule_node *node)
{
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *domain;
	isl_schedule_node *graft;

	ctx = isl_schedule_node_get_ctx(node);

	space = isl_space_set_alloc(ctx, 0, 0);
	space = isl_space_set_tuple_name(space, isl_dim_set, "init_device");
	domain = isl_union_set_from_set(isl_set_universe(space));
	graft = isl_schedule_node_from_domain(domain);

	node = isl_schedule_node_graft_before(node, graft);

	space = isl_space_set_alloc(ctx, 0, 0);
	space = isl_space_set_tuple_name(space, isl_dim_set, "clear_device");
	domain = isl_union_set_from_set(isl_set_universe(space));
	graft = isl_schedule_node_from_domain(domain);

	node = isl_schedule_node_graft_after(node, graft);

	return node;
}
