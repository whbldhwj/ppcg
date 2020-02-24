#include <isl/ctx.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/printer.h>
#include <isl/aff.h>

int main(){
  isl_ctx *ctx;
  isl_schedule *schedule;
  isl_ast_build *build;
  isl_ast_node *ast;
  isl_printer *p, *pd;
  isl_union_set *domain;
  isl_set *set;
  isl_map *map;
  isl_union_map *umap;
  const char *input_set = "{S[i,j]: 1<=i<=4 and 1<=j<=i and 1<=j<=2}";
  isl_schedule_node *graft, *node;
  isl_multi_union_pw_aff *mupa;

  ctx = isl_ctx_alloc();
  set = isl_set_read_from_str(ctx, input_set);
  map = isl_set_identity(isl_set_copy(set));
  domain = isl_union_set_from_set(set);
  schedule = isl_schedule_from_domain(domain);
  
  pd = isl_printer_to_file(ctx, stdout);
  pd = isl_printer_set_yaml_style(pd, ISL_YAML_STYLE_BLOCK);
  pd = isl_printer_print_schedule(pd, schedule);
  printf("\n");

  /* Insert a band node */
  map = isl_map_reset_tuple_id(map, isl_dim_out);
  umap = isl_union_map_from_map(map);
  mupa = isl_multi_union_pw_aff_from_union_map(umap);
  schedule = isl_schedule_insert_partial_schedule(schedule, mupa);

  pd = isl_printer_print_schedule(pd, schedule);
  printf("\n");

  node = isl_schedule_get_root(schedule);
  node = isl_schedule_node_child(node, 0);
  node = isl_schedule_node_band_member_set_ast_loop_type(node, 0, isl_ast_loop_unroll);
  node = isl_schedule_node_band_member_set_ast_loop_type(node, 1, isl_ast_loop_unroll);

  pd = isl_printer_print_schedule_node(pd, node);
  printf("\n");

  isl_schedule_free(schedule);
  schedule = isl_schedule_node_get_schedule(node);

  pd = isl_printer_print_schedule(pd, schedule);
  printf("\n");
 

  build = isl_ast_build_alloc(ctx);
  ast = isl_ast_build_node_from_schedule(build, schedule);
  p = isl_printer_to_file(ctx, stdout);
  p = isl_printer_set_output_format(p, ISL_FORMAT_C);
  p = isl_printer_print_ast_node(p, ast);
  printf("\n");
  p = isl_printer_free(p);

  return 0;
}
