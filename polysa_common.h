#ifndef _POLYSA_COMMON_H_
#define _POLYSA_COMMON_H_

#include <assert.h>
#include <limits.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/id.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "ppcg.h"
#include "schedule.h"
#include "gpu.h"
#include "util.h"
#include "polysa_tree.h"
#include "polysa_utilities.h"
#include "polysa_array_tile.h"

enum polysa_group_access_type {
  POLYSA_ACCESS_GLOBAL,
  POLYSA_ACCESS_LOCAL,
  POLYSA_ACCESS_SHARED,
  POLYSA_ACCESS_PRIVATE
};

enum polysa_kernel_stmt_type {
  POLYSA_KERNEL_STMT_COPY,
  POLYSA_KERNEL_STMT_DOMAIN,
  POLYSA_KERNEL_STMT_SYNC,
  POLYSA_KERNEL_STMT_IO,
  POLYSA_KERNEL_STMT_IO_TRANSFER,
  POLYSA_KERNEL_STMT_IO_TRANSFER_BUF,
  POLYSA_KERNEL_STMT_IO_DRAM,
  POLYSA_KERNEL_STMT_FIFO_DECL,
  POLYSA_KERNEL_STMT_MODULE_CALL,
  POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_TRANS,
  POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_TRANS,
  POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTER_INTRA,
  POLYSA_KERNEL_STMT_IO_MODULE_CALL_INTRA_INTER,
  POLYSA_KERNEL_STMT_IO_MODULE_CALL_STATE_HANDLE
};

enum polysa_dep_type {
  POLYSA_DEP_RAW,
  POLYSA_DEP_RAR,
  POLYSA_DEP_WAR,
  POLYSA_DEP_WAW,
  POLYSA_DEP_UNKNOWN
};

enum polysa_io_type {
  POLYSA_INT_IO,
  POLYSA_EXT_IO,
  POLYSA_UNKNOWN_IO
};

enum polysa_io_dir {
  IO_IN,
  IO_OUT,
  IO_INOUT,
  IO_NULL,
  IO_UNKNOWN
};

enum polysa_module_type {
  PE_MODULE,
  IO_MODULE,
  DRAIN_MODULE
};

enum polysa_group_type {
  POLYSA_IO_GROUP,
  POLYSA_PE_GROUP,
  POLYSA_DRAIN_GROUP,
  POLYSA_UNKNOWN_GROUP
};

enum polysa_array_type {
  POLYSA_EXT_ARRAY,
  POLYSA_INT_ARRAY
};

enum platform {
  INTEL_HW,
  XILINX_HW
};

struct hls_info {
  FILE *host_c;          /* OpenCL host. */
  FILE *kernel_c;        /* Definition of hardware modules. */
  FILE *kernel_h;        /* Declaration of hardware modules. */

  FILE *top_gen_c;       /* Prints out the top module that connects the 
                            hardware modules. */
  FILE *top_gen_h;      
  enum platform target;
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

/* A sequence of "n" names of types.
 */
struct polysa_types {
  int n;
  char **name;
};

struct polysa_iter {
  char *name;
  isl_aff *lb;
  isl_aff *ub;
  int stride;
  char *ts_name;
};

/* Representation of a local variable in a kernel 
 */
struct polysa_kernel_var {
  struct polysa_array_info *array;
  enum polysa_group_access_type type;
  char *name;
  isl_vec *size;
  int n_lane;
  /* Array partition factors */
  int n_part; 
};

struct polysa_kernel {
  isl_ctx *ctx;
  isl_schedule *schedule;
  struct ppcg_scop *scop;

  struct polysa_prog *prog;
  struct ppcg_options *options;

  int n_sa_dim;
  int sa_dim[3];
  int array_part_w;
  int space_w;
  int time_w;
  int simd_w;

  int type; // POLYSA_SA_TYPE_ASYNC | POLYSA_SA_TYPE_SYNC

  isl_multi_pw_aff *sa_grid_size;

  /* User specified (array_part/latency_hiding/simd) sizes for each kernel. */
  isl_union_map *sizes;

  /* Effectively used (array_part/latency_hiding/simd) sizes for each kernel. */
  isl_union_map *used_sizes;

  /* Identifier of the kernel. */
  int id;

  /* The spaces of teh statement domains that form the core computation of the 
   * kernel. 
   */
  isl_union_set *core;

  /* the set of possibly accessed outer array elements. */  
  isl_union_set *arrays;
  /* n_array is the total number of arrays in the input program and also
   * the number of element in the "array".
   * array contains information about each array that is local to the current
   * kernel. If an array is not used in a kernel, then the corresponding 
   * entry does not contain any information.
   */
  int n_array;
  struct polysa_local_array_info *array;

  /* copy_schdule corresponds to the schedule dimensions of the 
   * (tiled) schedule for this kernel that have been taken into account
   * for computing private/shared memory tiles.
   * copy_schedule_dim is the dimension of this schedule. 
   */
  isl_union_pw_multi_aff *copy_schedule;
  int copy_schedule_dim;

  /* space is the schedule space of the AST context. That is, it represents
   * the loops of the generated host code containing the kernel launch. 
   */
  isl_space *space;

  isl_ast_node *tree;

  /* local variables in a kernel. */
  int n_var; 
  struct polysa_kernel_var *var;

  /* contains the list of block identifiers for this kernel. */
  isl_id_list *block_ids;
  /* contains the list of thread identifiers for this kernel. */
  isl_id_list *thread_ids;
  /* contains the list of PE identifers for this kernel. */
  isl_id_list *pe_ids;

  /* contains constraints on the domain elements in the kernel
   * that encode the mapping to PE identifiers, where the PE identifiers
   * are represented by "space_w" parameters with the names as the elements
   * of "pe_ids".
   */
  isl_union_set *pe_filter;

  /* The first n_grid elements of grid_dim represent the specified size of 
   * the grid.
   * The first n_block elements of block_dim represent the specified or 
   * effective size of tghe block.
   * Note that in the input file, the sizes of the grid and the blocks 
   * are specified in the order x, y, z, but internally, the sizes 
   * are stored in reverse order, so that the last elments always referes
   * to the x dimension.
   *
   * grid_size reflects the effective grid size.
   * grid_size_expr contains a corresponding access AST expression, built within
   * the context where the launch appears.
   */
  int n_grid;
  int n_block;
  int grid_dim[2];
  int block_dim[3];

  isl_multi_pw_aff *grid_size;
  isl_ast_expr *grid_size_expr;

  /* contains the values of the parameters and outer schedule dimensions
   * for which any statement instance in this kernel needs to be executed.
   */
  isl_set *context;

  /* contraction maps those original statement instances to the statement
   * instances that are active at the point in the schedule tree where 
   * the kernel is created.
   */
  isl_union_pw_multi_aff *contraction;
  /* contains the original statement instances,
   * i.e., those that appear in the domains of access relations, 
   * that are involved in the kernel. 
   */
  isl_union_set *expanded_domain;
  isl_union_set *domain;

  isl_set *host_domain;

  int single_statement;
};

struct polysa_io_info {
  enum polysa_io_type io_type;
//  isl_basic_map *dep;
  struct polysa_dep *dep;
  isl_vec *dir;
  /* Old data transfer direction before interior I/O elimination */
  isl_vec *old_dir;
};

/* An access to an outer array element or an iterator.
 * Accesses to iterators have an access relation that maps to an unnamed space.
 * An access may be both read and write.
 * If the access relation is empty, then the output dimension may
 * not be equal to the dimension of the corresponding array.
 */
struct polysa_stmt_access {
	/* Access reads elements */
	int read;
	/* Access writes elements */
	int write;
	/* All writes are definite writes. */
	int exact_write;
	/* Is a single, fixed element being accessed? */
	isl_bool fixed_element;
	/* The number of index expressions specified in the access. */
	int n_index;

	/* May access relation */
	isl_map *access;
	/* May access relation with as domain a mapping from iteration domain
	 * to a reference identifier.
	 */
	isl_map *tagged_access;
	/* The reference id of the corresponding pet_expr. */
	isl_id *ref_id;

  /* PolySA extended */
  struct polysa_io_info **io_info;
  int n_io_info;
  /* Indicates if layout transformation is required for SIMD */
  int layout_trans;
  /* Indicates which array dimension should be permuted innmermost for SIMD */
  int simd_dim;
  /* Indicates the stride pattern under the SIMD loop.
   * Default value as -1. 0 if stride-0 and 1 if stride-1 */
  int simd_stride;
  /* PolySA extended */

	struct polysa_stmt_access *next;  
};

/* Internal data structure for extract_access.
 * "next_access" points to the end of a linked list that is extended
 * by extract_access.
 * "single_expression" is set if the access expressions belong to
 * an expression statement (i.e., a statement without internal control).
 * "any_to_outer" maps all intermediate arrays to their outer arrays.
 */
struct ppcg_extract_access_data {
	struct polysa_stmt_access **next_access;
	int single_expression;
	isl_union_map *any_to_outer;
};

/* A representation of a user statement.
 * "stmt" points to the corresponding pet statement.
 * "id" is the identifier of the instance set of the statement.
 * "accesses" is a linked list of accesses performed by the statement.
 * If the statement has been killed, i.e., if it will not be scheduled,
 * then this linked list may be empty even if the actual statement does
 * perform accesses.
 */
struct polysa_stmt {
  isl_id *id;
  struct pet_stmt *stmt;

  struct polysa_stmt_access *accesses;
};

/* Represents an outer array possibly accessed by a gpu_prog.
 */
struct polysa_array_info {
	/* The array data space. */
	isl_space *space;
	/* Element type. */
	char *type;
	/* Element size. */
	int size;
	/* Name of the array. */
	char *name;
	/* Declared extent of original array. */
	isl_set *declared_extent;
	/* AST expression for declared size of original array. */
	isl_ast_expr *declared_size;
	/* Extent of the array that needs to be copied. */
	isl_set *extent;
	/* Number of indices. */
	unsigned n_index;
	/* For each index, a bound on "extent" in that direction. */
	isl_multi_pw_aff *bound;
	/* The corresponding access AST expression, if the array needs
	 * to be allocated on the device.
	 */
	isl_ast_expr *bound_expr;

	/* All references to this array; point to elements of a linked list. */
	int n_ref;
	struct polysa_stmt_access **refs;

	/* Is this array accessed at all by the program? */
	int accessed;

	/* Is this a scalar that is read-only within the entire program? */
	int read_only_scalar;

	/* Are the elements of the array structures? */
	int has_compound_element;

	/* Are the elements only accessed through constant index expressions? */
	int only_fixed_element;

	/* Is the array local to the scop? */
	int local;
	/* Is the array local and should it be declared on the host? */
	int declare_local;

	/* Is the corresponding global device memory accessed in any way? */
	int global;

	/* Should the array be linearized? */
	int linearize;

	/* Order dependences on this array.
	 * Only used if live_range_reordering option is set.
	 * It is set to NULL otherwise.
	 */
	isl_union_map *dep_order;
};

struct polysa_io_buffer {
  /* The local buffer tile, NULL if none. */
  struct polysa_array_tile *tile;
  /* The buffer is located at io_L"level". */
  int level;
  /* The data pack factor */
  int n_lane;
};

/* A group of array references in a kernel that should be handled together.
 * If private_tile is not NULL, then it is mapped to registers.
 * Otherwise, if shared_tile is not NULL, it is mapped to shared memory.
 * Otherwise, it is accessed from global memory.
 * Note that if both private_tile and shared_tile are set, then shared_tile
 * is only used inside group_common_shared_memory_tile.
 */
struct polysa_array_ref_group {
  /* The references in this group access this local array. */
  struct polysa_local_array_info *local_array;
  /* This is the corresponding array. */
  struct polysa_array_info *array;
  /* Position of this group in the list of reference group of array. */
  int nr;

	/* The following fields are use during the construction of the groups.
	 * access is the combined access relation relative to the private
	 * memory tiling.  In particular, the domain of the map corresponds
	 * to the first thread_depth dimensions of the kernel schedule.
	 * write is set if any access in the group is a write.
	 * exact_write is set if all writes are definite writes.
	 * slice is set if there is at least one access in the group
	 * that refers to more than one element
	 * "min_depth" is the minimum of the tile depths and thread_depth.
	 */
	isl_map *access;
	int write;
	int exact_write;
	int slice;
	int min_depth;

	/* The shared memory tile, NULL if none. */
	struct polysa_array_tile *shared_tile;

	/* The private memory tile, NULL if none. */
	struct polysa_array_tile *private_tile;

  /* The local memory tile, NULL if none. */
  struct polysa_array_tile *local_tile;

	/* References in this group; point to elements of a linked list. */
	int n_ref;
	struct polysa_stmt_access **refs;  

  /* PolySA Extended */

  /* I/O buffers inserted at each IO level */
  struct polysa_io_buffer **io_buffers;   
  int n_io_buffer;
  /* I/O type: interior/exterior I/O */
  enum polysa_io_type io_type;
  /* I/O direction at the PE level */
  isl_vec *dir;
  /* Group type: I/O/drain/PE group */
  enum polysa_group_type group_type;
  /* I/O direction at the PE level */
  enum polysa_io_dir pe_io_dir;
  /* I/O direction at the array level */
  enum polysa_io_dir array_io_dir;
  /* Maps PE identifiers to I/O identifiers */
  isl_multi_aff *io_trans; /* pe ids -> io ids */ // TODO
  isl_multi_aff *io_L1_trans; /* pe ids -> L1 io ids */
//  /* Maps PE identifiers to I/O identifiers */
//  isl_mat *io_trans_mat; // TODO
  /* AST expression maps L1 I/O identifiers to PE identifiers */
  isl_ast_expr *io_pe_expr; /* io ids -> pe ids */ // TODO
  isl_ast_expr *io_L1_pe_expr; /* L1 io ids -> pe ids */ // TODO
  /* I/O schedule */
  isl_schedule *io_schedule;
  isl_schedule *io_L1_schedule;
  /* Number of I/O levels */
  int io_level;
  /* Dims of space band */
  int space_dim;
  /* Data pack factor inside PEs */
  int n_lane;
  /* Copy schedule for PE group */
  int copy_schedule_dim;
  isl_union_pw_multi_aff *copy_schedule;

  /* PolySA Extended */
};

struct polysa_array_ref_group_pair {
  struct polysa_array_ref_group *local_group; 
  struct polysa_array_ref_group *io_group;
  struct polysa_array_tile *local_tile; /* Compute the local tile */
  int in_use;
};

/* Represents an outer array accessed by a polysa_kernel, localized
 * to the context of this kernel.
 *
 * "array" points to the corresponding array in the polysa_prog.
 * The "n_group" "groups" are the reference groups associated to the array.
 * If "force_private" is set, then the array (in practice a scalar)
 * must be mapped to a register.
 * "global" is set if the global device memory corresponding
 * to this array is accessed by the kernel.
 * "bound" is equal to array->bound specialized to the current kernel.
 * "bound_expr" is the corresponding access AST expression.
 */
struct polysa_local_array_info {
	struct polysa_array_info *array;

  /* PE groups */
	int n_pe_group;
	struct polysa_array_ref_group **pe_groups;

  /* IO groups */
  int n_io_group;
  struct polysa_array_ref_group **io_groups;

  /* Drain groups */
  struct polysa_array_ref_group *drain_group;

  /* Default groups */
  int n_group;
  struct polysa_array_ref_group **groups;

  enum polysa_array_type array_type;
  int n_lane;

	int force_private;
	int global;

	unsigned n_index;
	isl_multi_pw_aff *bound;
	isl_ast_expr *bound_expr;
};

/* "read" and "write" contain the original access relations, possibly 
 * involving member accesses.
 * 
 * The elements of "array", as well as the ranges of "copy_in" and "copy_out"
 * only refer to the outer arrays of any possible member accesses.
 */
struct polysa_prog {
  isl_ctx *ctx;

  struct ppcg_scop *scop;

  /* Set of parameter values */
  isl_set *context;

  /* All potential read accesses in the entire program */
  isl_union_map *read;

  /* All potential write accesses in the entire program */
  isl_union_map *may_write;
  /* All definite write accesses in the entire program */
  isl_union_map *must_write;
  /* All tagged definite kills in the entire program */
  isl_union_map *tagged_must_kill;

  /* The set of inner array elements that may be preserved. */
  isl_union_set *may_persist;

  /* A mapping from all innermost arrays to their outer arrays. */
  isl_union_map *to_outer;
  /* A mapping from all the outer arrays to all corresponding inner arrays */
  isl_union_map *to_inner;
	/* A mapping from all intermediate arrays to their outer arrays,
	 * including an identity mapping from the anonymous 1D space to itself.
	 */
	isl_union_map *any_to_outer;

	/* Order dependences on non-scalars. */
	isl_union_map *array_order;

	/* Array of statements */
	int n_stmts;
	struct polysa_stmt *stmts;

	int n_array;
	struct polysa_array_info *array;  
};

struct polysa_hw_top_module {
  int n_fifo_decls;
  int n_module_calls;
  isl_schedule **fifo_decl_scheds;
  isl_schedule **module_call_scheds;
  isl_ast_node **fifo_decl_trees;
  isl_ast_node **module_call_trees;
  
  /* Wrapped AST */
  int n_fifo_decl_wrapped;
  int n_module_call_wrapped;
  isl_ast_node **fifo_decl_wrapped_trees;
  isl_ast_node **module_call_wrapped_trees;

  int n_hw_modules;
  struct polysa_hw_module **hw_modules;
  struct polysa_kernel *kernel;
};

struct polysa_hw_module {
  enum polysa_module_type type;
  /* Module name */
  char *name;

  isl_id_list *inst_ids;
  int n_var;
  struct polysa_kernel_var *var;

  /* Module function schedule */
  isl_schedule *sched;

  /* Module function AST */
  isl_ast_node *tree;
  isl_ast_node *device_tree;

  /* Array reference group for I/O or drain module */
  struct polysa_array_ref_group **io_groups;
  int n_io_group;

  /* I/O module level */
  int level;
  /* I/O module copy-in/out */
  int in;
  /* Connect to external memory */
  int to_mem; 
  /* Connect to PE */
  int to_pe;
  /* Contains buffer */
  int is_buffer;
  /* Filter module */
  int is_filter;

  /* Module function schedule for buffer_filter modules */
  isl_schedule *outer_sched; /* Outer loops */
  isl_schedule *inter_sched; /* Inter transfer */
  isl_schedule *intra_sched; /* Intra transfer */
  
  isl_schedule *boundary_outer_sched;
  isl_schedule *boundary_inter_sched;

  isl_space *inter_space;
  isl_space *intra_space;
  isl_space *space;

  isl_ast_node *inter_tree;
  isl_ast_node *intra_tree;
  
  isl_ast_node *boundary_outer_tree;
  isl_ast_node *boundary_inter_tree;

  /* Module function schedule for filter modules at the boundary */
  isl_schedule *boundary_sched;
  isl_ast_node *boundary_tree;
  int boundary;

  int double_buffer;

  /* Generate credit control */
  int credit;

  /* Data pack factor */
  int data_pack_inter;
  int data_pack_intra;

  struct polysa_kernel *kernel;
};

struct polysa_gen {
  isl_ctx *ctx;
  struct ppcg_options *options;

  /* Callback for printing of AST in appropriate format. */
  __isl_give isl_printer *(*print)(__isl_take isl_printer *p,
    struct polysa_prog *prog, __isl_keep isl_ast_node *tree, 
    struct polysa_hw_module **modules, int n_modules,
    struct polysa_hw_top_module *top_module,
    struct polysa_types *types, void *user);
  void *print_user;

  struct polysa_prog *prog;  
  struct polysa_kernel *kernel;
  /* The default AST */
  isl_ast_node *tree;

  /* The default schedule */
  isl_schedule *schedule;

  /* The SA module schedule */
  struct polysa_hw_module **hw_modules;
  int n_hw_modules;
  struct polysa_hw_top_module *hw_top_module;

  /* The sequence of types for which a definition has been printed. */
  struct polysa_types types;

  /* User specified tile sizes for each kernel. */
  isl_union_map *sizes;

  /* Effectively used tile sizes for each kernel. */
  isl_union_map *used_sizes;

  /* Identifier of the next kernel. */
  int kernel_id;
};

/* Representation of special statements, in particular copy statements
 * ,__syncthreads statements, and I/O statements, inside a kernel.
 *
 * type represents the kind of statement
 *
 *
 * for polysa_kernel_copy statements we have
 *
 * read is set if the statement should copy data from global memory
 * to shared memory or registers.
 *
 * index expresses an access to the array element that needs to be copied
 * local_index expresses the corresponding element in the tile
 *
 * array refers to the original array being copied
 * local_array is a pointer to the appropriate element in the "array"
 *	array of the polysa_kernel to which this copy access belongs
 *
 *
 * for polysa_kernel_domain statements we have
 *
 * stmt is the corresponding input statement
 *
 * n_access is the number of accesses in stmt
 * access is an array of local information about the accesses
 *
 * for polysa_kernel_io statements we have
 *
 * in is set if the statement should read data from fifo 
 * to local array or registers.
 *
 * local_index expresses the corresponding element in the tile
 *
 * array refers to the original array being transferred
 * local_array is a pointer to the appropriate element in the "array"
 *  array of the polysa_kernel to which this copy access belongs
 */
struct polysa_kernel_stmt {
	enum polysa_kernel_stmt_type type;

	union {
		struct {
			int read;
			isl_ast_expr *index;
			isl_ast_expr *local_index;
			struct polysa_array_info *array;
			struct polysa_local_array_info *local_array;
		} c;
		struct {
			struct polysa_stmt *stmt;
			isl_id_to_ast_expr *ref2expr;
		} d;
    struct {
      int in;
      int buf;
      int filter;
      int boundary;
      char *fifo_name;
      char *fifo_type;
      int filter_sched_depth;
      int filter_param_id;
      int data_pack;
      int reg;
      int nxt_data_pack;
      isl_ast_expr *local_index;
      isl_ast_expr *index;
      struct polysa_array_info *array;
      struct polysa_local_array_info *local_array;
      struct polysa_array_ref_group *group;
      struct polysa_hw_module *module;
    } i;
    struct {
      struct polysa_hw_module *module;
      struct polysa_array_ref_group *group;
      int boundary;
      int upper;
      int lower;
    } m;
    struct {
      struct polysa_hw_module *module;
      int boundary;
    } f;
	} u;
};


struct polysa_acc {
  isl_map *tagged_map;
  isl_map *map;
  isl_space *id;

  int rw; // 0 - read 1 - write
};

struct polysa_node_band_prop {
  int permutable;
  int *coincident;
  enum polysa_loop_type *pe_opt;
  enum polysa_loop_type *space_time;
  int n_member;
  isl_multi_union_pw_aff *mupa;
};

struct polysa_ast_node_userinfo {
  int is_pipeline;
  int is_unroll;
};

/* Band node related functions */
__isl_give isl_multi_val *construct_band_tile_sizes(
  __isl_keep isl_schedule_node *node, int *tile_size);
struct polysa_node_band_prop *extract_node_band_prop(__isl_keep isl_schedule_node *node);
struct polysa_node_band_prop *polysa_node_band_prop_free(__isl_take struct polysa_node_band_prop *prop);
isl_bool is_permutable_node(__isl_keep isl_schedule_node *node);
isl_bool has_single_permutable_node(__isl_keep isl_schedule *schedule);
isl_bool is_dep_uniform_at_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool is_dep_uniform(__isl_keep isl_basic_map *bmap, void *user);
isl_bool is_dep_uniform_wrap(__isl_keep isl_map *map, void *user);
isl_bool uniform_dep_check(__isl_keep isl_schedule *schedule, struct ppcg_scop *scop);
__isl_give isl_vec *get_dep_dis_at_schedule(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule *schedule);
__isl_give isl_vec *get_dep_dis_at_node(__isl_keep isl_basic_map *dep, __isl_keep isl_schedule_node *band);
__isl_give isl_schedule *loop_interchange_at_node(__isl_take isl_schedule_node *node, isl_size level1, isl_size level2);
__isl_give isl_schedule_node *get_outermost_permutable_node(__isl_keep isl_schedule *schedule);
__isl_give isl_schedule_node *get_innermost_permutable_node(__isl_keep isl_schedule *schedule);
__isl_give isl_schedule_node *tile_band(
  __isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_node *polysa_tile_band(
  __isl_take isl_schedule_node *node, __isl_keep int *sizes);	
__isl_give isl_schedule_node *clear_pe_opt_prop(
  __isl_take isl_schedule_node *node, void *user);
__isl_give isl_schedule_node *restore_node_band_prop(__isl_take isl_schedule_node *node, 
  __isl_take struct polysa_node_band_prop *prop);
__isl_give isl_schedule_node *polysa_node_interchange(__isl_take isl_schedule_node *node);
isl_bool no_permutable_node(isl_schedule_node *node, void *user);
isl_bool all_parallel_node(__isl_keep isl_schedule_node *node, void *user);
isl_bool isl_schedule_node_is_io_mark(__isl_keep isl_schedule_node *node, int io_level);
int is_node_under_simd(__isl_keep isl_schedule_node *node);
int is_node_under_latency(__isl_keep isl_schedule_node *node);

/* PolySA kernel related functions */
void *polysa_kernel_free(struct polysa_kernel *sa);
struct polysa_kernel *polysa_kernel_copy(struct polysa_kernel *sa);
struct polysa_kernel *polysa_kernel_from_schedule(__isl_take isl_schedule *schedule);
struct polysa_kernel *polysa_kernel_alloc(isl_ctx *ctx, struct ppcg_scop *scop);

/* Other PolySA structs */
void *polysa_acc_free(struct polysa_acc *acc);
__isl_null struct polysa_iter *polysa_iter_free(struct polysa_iter *iter);

/* PolySA dep related functions */
void *polysa_dep_free(__isl_take struct polysa_dep *dep);

/* Schedule related functions */
__isl_give isl_schedule *compute_schedule(struct polysa_gen *gen);
__isl_give isl_schedule *get_schedule(struct polysa_gen *gen);

/* PolySA array related functions */
isl_stat collect_array_info(struct polysa_prog *prog);
int polysa_array_is_read_only_scalar(struct polysa_array_info *array);
int polysa_array_is_scalar(struct polysa_array_info *array);
int polysa_kernel_requires_array_argument(struct polysa_kernel *kernel, int i);
struct polysa_array_ref_group *polysa_array_ref_group_free(
	struct polysa_array_ref_group *group);

/* PolySA stmts related functions */
struct polysa_stmt *extract_stmts(isl_ctx *ctx, struct ppcg_scop *scop,
  __isl_keep isl_union_map *any_to_outer);
void polysa_kernel_stmt_free(void *user);
struct polysa_stmt *find_stmt(struct polysa_prog *prog, __isl_keep isl_id *id);

/* PolySA prog related functions */
struct polysa_prog *polysa_prog_alloc(isl_ctx *ctx, struct ppcg_scop *scop);
void *polysa_prog_free(struct polysa_prog *prog);

/* PolySA hw module related functions */
struct polysa_hw_module *polysa_hw_module_alloc();
void *polysa_hw_module_free(struct polysa_hw_module *module);
struct polysa_hw_top_module *polysa_hw_top_module_alloc();
void *polysa_hw_top_module_free(struct polysa_hw_top_module *module);

/* PolySA ast node related functions */
struct polysa_ast_node_userinfo *alloc_ast_node_userinfo();

/* PolySA MISC */
int *read_hbm_tile_sizes(struct polysa_kernel *sa, int *tile_len);
__isl_give isl_set *extract_sa_sizes(__isl_keep isl_union_map *sizes,
    const char *type, int id);
int *read_array_part_tile_sizes(struct polysa_kernel *sa, int *tile_len);
int *read_latency_tile_sizes(struct polysa_kernel *sa, int *tile_len);
int *read_simd_tile_sizes(struct polysa_kernel *sa, int *tile_len);

#endif
