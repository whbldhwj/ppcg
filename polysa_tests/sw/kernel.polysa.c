#include "kernel.h"

void dsa_kernel(char alt[M], char ref[N], int H[M + 1][N + 1], int bt[M][N]) {
  int best_gap_h[M];
  int gap_size_h[M];
  int best_gap_v[N];
  int gap_size_v[N];
 
  for (int i = 0; i < M; i++)
    best_gap_h[i] = LOW_INIT_VALUE;
  for (int j = 0; j < N; j++)
    best_gap_v[j] = LOW_INIT_VALUE;
   for (int i = 0; i < M; i++)
    gap_size_h[i] = 0;
  for (int j = 0; j < N; j++)
    gap_size_v[j] = 0;  

  /* PPCG generated CPU code */
  
  {
    char alt_base;
    char ref_base;
    int sim_score;
    int H_left;
    int H_up;
    int H_leftup;
    int step_diag;
    int prev_gap;
    int step_down;
    int kd;
    int step_right;
    int ki;
    int bt_tmp;
    int sw_tmp;
    for (int c0 = 0; c0 <= 15; c0 += 1)
      for (int c1 = 0; c1 <= 7; c1 += 1) {
        alt_base = alt[c0];
        ref_base = ref[c1];
        sim_score = ((alt_base == ref_base) ? 200 : (-150));
        H_left = H[c0 + 1][c1];
        H_up = H[c0][c1 + 1];
        H_leftup = H[c0][c1];
        step_diag = (H_leftup + sim_score);
        step_diag = (((-100000000) > step_diag) ? (-100000000) : step_diag);
        prev_gap = (H_up + (-260));
        best_gap_v[c1] += (-11);
        if (prev_gap > best_gap_v[c1]) {
          best_gap_v[c1] = prev_gap;
          gap_size_v[c1] = 1;
        } else {
          gap_size_v[c1]++;
        }
        step_down = best_gap_v[c1];
        step_down = (((-100000000) > step_down) ? (-100000000) : step_down);
        kd = gap_size_v[c1];
        prev_gap = (H_left + (-260));
        best_gap_h[c0] += (-11);
        if (prev_gap > best_gap_h[c0]) {
          best_gap_h[c0] = prev_gap;
          gap_size_h[c0] = 1;
        } else {
          gap_size_h[c0]++;
        }
        step_right = best_gap_h[c0];
        step_right = (((-100000000) > step_right) ? (-100000000) : step_right);
        ki = gap_size_h[c0];
        sw_tmp = ((step_diag > step_down) ? step_diag : step_down);
        sw_tmp = ((sw_tmp > step_right) ? sw_tmp : step_right);
        bt_tmp = kd;
        bt_tmp = ((sw_tmp == step_right) ? (-ki) : bt_tmp);
        bt_tmp = ((sw_tmp == step_diag) ? 0 : bt_tmp);
        H[c0 + 1][c1 + 1] = sw_tmp;
        bt[c0][c1] = bt_tmp;
      }
  }
}
