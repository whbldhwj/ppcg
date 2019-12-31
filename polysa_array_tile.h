#ifndef _POLYSA_ARRAY_TILE_H_
#define _POLYSA_ARRAY_TILE_H_

#include <isl/aff_type.h>
#include <isl/map_type.h>
#include <isl/val.h>

/* The current index is such that if you add "shift",
 * then the result is always a multiple of "stride",
 * where "stride" may be equal to 1.
 * Let D represent the initial tile->depth dimensions of the computed schedule.
 * The spaces of "lb" and "shift" are of the form
 *
 *	D -> [b]
 */
struct polysa_array_bound {
	isl_val *size;
	isl_aff *lb;

	isl_val *stride;
	isl_aff *shift;
};

/* A tile of an outer array.
 *
 * requires_unroll is set if the schedule dimensions that are mapped
 * to threads need to be unrolled for this (private) tile to be used.
 *
 * "depth" reflects the number of schedule dimensions that affect the tile.
 * The copying into and/or out of the tile is performed at that depth.
 *
 * n is the dimension of the array.
 * bound is an array of size "n" representing the lower bound
 *	and size for each index.
 *
 * tiling maps a tile in the global array to the corresponding
 * local memory tile and is of the form
 *
 *	{ [D[i] -> A[a]] -> T[(a + shift(i))/stride - lb(i)] }
 *
 * where D represents the initial "depth" dimensions
 * of the computed schedule.
 */
struct polysa_array_tile {
	isl_ctx *ctx;
	int requires_unroll;
	int depth;
	int n;
	struct polysa_array_bound *bound;
	isl_multi_aff *tiling;
};

struct polysa_array_tile *polysa_array_tile_free(struct polysa_array_tile *tile);
struct polysa_array_tile *polysa_array_tile_create(isl_ctx *ctx, int n_index);
__isl_give isl_val *polysa_array_tile_size(struct polysa_array_tile *tile);

#endif
