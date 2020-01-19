#include "polysa_utilities.h"
#include <assert.h>

isl_stat unionize_pw_aff_space(__isl_take isl_pw_aff *pa, void *user)
{
  isl_space *space = user;
  isl_space *space_i = isl_pw_aff_get_space(pa);
  
  isl_pw_aff_free(pa);
}

__isl_give isl_union_map *extract_sizes_from_str(isl_ctx *ctx, const char *str)
{
  if (!str)
    return NULL;
  return isl_union_map_read_from_str(ctx, str);
}

isl_stat concat_basic_map(__isl_take isl_map *el, void *user) 
{
  isl_basic_map_list **bmap_list = (isl_basic_map_list **)(user);
  isl_basic_map_list *bmap_list_sub = isl_map_get_basic_map_list(el);
  if (!(*bmap_list)) {
    *bmap_list = bmap_list_sub;
  } else {
    *bmap_list = isl_basic_map_list_concat(*bmap_list, bmap_list_sub);
  }

  isl_map_free(el);
  return isl_stat_ok;
}

__isl_give isl_basic_map_list *isl_union_map_get_basic_map_list(__isl_keep isl_union_map *umap)
{
  isl_map_list *map_list = isl_union_map_get_map_list(umap);
  isl_basic_map_list *bmap_list = NULL;
  isl_map_list_foreach(map_list, &concat_basic_map, &bmap_list);

  isl_map_list_free(map_list);
  return bmap_list;
}

static isl_stat acc_n_basic_map(__isl_take isl_map *el, void *user)
{
  isl_size *n = (isl_size *)(user);
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(el);
  *n = *n + isl_basic_map_list_n_basic_map(bmap_list);

//  // debug
//  isl_printer *printer = isl_printer_to_file(isl_map_get_ctx(el), stdout);
//  isl_printer_print_map(printer, el);
//  printf("\n");  
//  printf("%d\n", *n);
//  // debug

  isl_map_free(el);
  isl_basic_map_list_free(bmap_list);
  return isl_stat_ok;
}

isl_size isl_union_map_n_basic_map(__isl_keep isl_union_map *umap)
{
  isl_size n = 0;
  isl_map_list *map_list = isl_union_map_get_map_list(umap);
  isl_map_list_foreach(map_list, &acc_n_basic_map, &n);

  isl_map_list_free(map_list);

  return n;
}

__isl_give isl_basic_map *isl_basic_map_from_map(__isl_take isl_map *map)
{
  if (!map)
    return NULL;

  assert(isl_map_n_basic_map(map) == 1);
  isl_basic_map_list *bmap_list = isl_map_get_basic_map_list(map);
  isl_map_free(map);

  isl_basic_map *bmap = isl_basic_map_list_get_basic_map(bmap_list, 0);
  isl_basic_map_list_free(bmap_list);

  return bmap;
}

void print_mat(FILE *fp, __isl_keep isl_mat *mat) 
{
  isl_printer *printer = isl_printer_to_file(isl_mat_get_ctx(mat), fp);
  for (int i = 0; i < isl_mat_rows(mat); i++) {
    for (int j = 0; j < isl_mat_cols(mat); j++) {
      isl_printer_print_val(printer, isl_mat_get_element_val(mat, i, j));
      fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }
  isl_printer_free(printer);
}

/* Compare the two vectors, return 0 if equal.
 */
int isl_vec_cmp(__isl_keep isl_vec *vec1, __isl_keep isl_vec *vec2)
{
  if (isl_vec_size(vec1) != isl_vec_size(vec2))
    return 1;

  // debug
  isl_printer *p = isl_printer_to_file(isl_vec_get_ctx(vec1), stdout);
  p = isl_printer_print_vec(p, vec1);
  printf("\n");
  p = isl_printer_print_vec(p, vec2);
  printf("\n");
  // debug

  for (int i = 0; i < isl_vec_size(vec1); i++) {
    if (isl_vec_cmp_element(vec1, vec2, i))
      return 1;
  }

  return 0;
}
