#include "kernel.h"

void dsa_kernel(char alt[M], char ref[N], int H[M + 1][N + 1], int bt[M][N]) {
  char alt_ext[M][N];
  char ref_ext[M][N];
  int sim_score_ext[M][N];
  int H_ext[M][N];

  int step_diag_ext[M][N];
 
  int best_gap_v_ext[M][N];
  int gap_size_v_ext[M][N];
  int step_down_ext[M][N];
  int kd_ext[M][N];
  
  int best_gap_h_ext[M][N];
  int gap_size_h_ext[M][N];
  int step_right_ext[M][N];
  int ki_ext[M][N];

  int sw_tmp1_ext[M][N];
  int sw_tmp2_ext[M][N];
  int bt_tmp1_ext[M][N];
  int bt_tmp2_ext[M][N];

  /* PPCG generated CPU code */
  
  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
  {
    for (int c0 = 0; c0 <= 7; c0 += 1)
      best_gap_v_ext[0][c0] = (-260);
    for (int c0 = 0; c0 <= 15; c0 += 1)
      alt_ext[c0][0] = alt[c0];
    for (int c0 = 0; c0 <= 7; c0 += 1)
      gap_size_v_ext[0][c0] = 1;
    for (int c0 = 0; c0 <= 7; c0 += 1)
      ref_ext[0][c0] = ref[c0];
    for (int c0 = 0; c0 <= 15; c0 += 1)
      for (int c1 = 1; c1 <= 7; c1 += 1)
        alt_ext[c0][c1] = alt_ext[c0][c1 - 1];
    for (int c0 = 0; c0 <= 7; c0 += 1)
      for (int c1 = 1; c1 <= 15; c1 += 1)
        ref_ext[c1][c0] = ref_ext[c1 - 1][c0];
    for (int c0 = 0; c0 <= 15; c0 += 1)
      for (int c1 = 0; c1 <= 7; c1 += 1) {
        sim_score_ext[c0][c1] = ((alt_ext[c0][c1] == ref_ext[c0][c1]) ? 200 : (-150));
        if (c0 == 0 || c1 == 0)
          step_diag_ext[c0][c1] = (((-100000000) > sim_score_ext[c0][c1]) ? (-100000000) : sim_score_ext[c0][c1]);
      }
    for (int c0 = 0; c0 <= 15; c0 += 1)
      best_gap_h_ext[c0][0] = (-260);
    for (int c0 = 0; c0 <= 15; c0 += 1)
      gap_size_h_ext[c0][0] = 1;
    for (int c0 = 0; c0 <= 22; c0 += 1)
      for (int c1 = ppcg_max(0, c0 - 7); c1 <= ppcg_min(15, c0); c1 += 1) {
        if (c0 >= c1 + 1)
          if ((H_ext[c1][c0 - c1 - 1] + (-260)) > (best_gap_h_ext[c1][c0 - c1 - 1] + (-11))) {
            best_gap_h_ext[c1][c0 - c1] = (H_ext[c1][c0 - c1 - 1] + (-260));
            gap_size_h_ext[c1][c0 - c1] = 1;
          } else {
            best_gap_h_ext[c1][c0 - c1] = (best_gap_h_ext[c1][c0 - c1 - 1] + (-11));
            gap_size_h_ext[c1][c0 - c1] = (gap_size_h_ext[c1][c0 - c1 - 1] + 1);
          }
        if (c1 >= 1)
          if ((H_ext[c1 - 1][c0 - c1] + (-260)) > (best_gap_v_ext[c1 - 1][c0 - c1] + (-11))) {
            best_gap_v_ext[c1][c0 - c1] = (H_ext[c1 - 1][c0 - c1] + (-260));
            gap_size_v_ext[c1][c0 - c1] = 1;
          } else {
            best_gap_v_ext[c1][c0 - c1] = (best_gap_v_ext[c1 - 1][c0 - c1] + (-11));
            gap_size_v_ext[c1][c0 - c1] = (gap_size_v_ext[c1 - 1][c0 - c1] + 1);
          }
        kd_ext[c1][c0 - c1] = gap_size_v_ext[c1][c0 - c1];
        step_down_ext[c1][c0 - c1] = (((-100000000) > best_gap_v_ext[c1][c0 - c1]) ? (-100000000) : best_gap_v_ext[c1][c0 - c1]);
        if (c1 >= 1 && c0 >= c1 + 1)
          step_diag_ext[c1][c0 - c1] = (((-100000000) > (H_ext[c1 - 1][c0 - c1 - 1] + sim_score_ext[c1][c0 - c1])) ? (-100000000) : (H_ext[c1 - 1][c0 - c1 - 1] + sim_score_ext[c1][c0 - c1]));
        step_right_ext[c1][c0 - c1] = (((-100000000) > best_gap_h_ext[c1][c0 - c1]) ? (-100000000) : best_gap_h_ext[c1][c0 - c1]);
        ki_ext[c1][c0 - c1] = gap_size_h_ext[c1][c0 - c1];
        sw_tmp1_ext[c1][c0 - c1] = ((step_diag_ext[c1][c0 - c1] > step_down_ext[c1][c0 - c1]) ? step_diag_ext[c1][c0 - c1] : step_down_ext[c1][c0 - c1]);
        sw_tmp2_ext[c1][c0 - c1] = ((sw_tmp1_ext[c1][c0 - c1] > step_right_ext[c1][c0 - c1]) ? sw_tmp1_ext[c1][c0 - c1] : step_right_ext[c1][c0 - c1]);
        H_ext[c1][c0 - c1] = sw_tmp2_ext[c1][c0 - c1];
        H[c1 + 1][c0 - c1 + 1] = H_ext[c1][c0 - c1];
        bt_tmp1_ext[c1][c0 - c1] = ((H_ext[c1][c0 - c1] == step_right_ext[c1][c0 - c1]) ? (-ki_ext[c1][c0 - c1]) : kd_ext[c1][c0 - c1]);
        bt_tmp2_ext[c1][c0 - c1] = ((H_ext[c1][c0 - c1] == step_diag_ext[c1][c0 - c1]) ? 0 : bt_tmp1_ext[c1][c0 - c1]);
        bt[c1][c0 - c1] = bt_tmp2_ext[c1][c0 - c1];
      }
  }
}
