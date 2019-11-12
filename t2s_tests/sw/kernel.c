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

#pragma scop
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      char alt_base = alt[i];
      char ref_base = ref[j];
      int sim_score = (alt_base == ref_base)? P_W_MATCH: P_W_MISMATCH;
      int H_left = H[(i + 1)][(j + 1) - 1];
      int H_up = H[(i + 1) - 1][(j + 1)];
      int H_leftup = H[(i + 1) - 1][(j + 1) - 1];

      int step_diag = H_leftup + sim_score;
      step_diag = (MATRIX_MIN_CUTOFF > step_diag)? MATRIX_MIN_CUTOFF: step_diag;
      int prev_gap = H_up + P_W_OPEN;
      best_gap_v[j] += P_W_EXTEND;
      if (prev_gap > best_gap_v[j]) {
        best_gap_v[j] = prev_gap;
        gap_size_v[j] = 1;
      } else {
        gap_size_v[j]++;
      }

      int step_down = best_gap_v[j];
      step_down = (MATRIX_MIN_CUTOFF > step_down)? MATRIX_MIN_CUTOFF: step_down;
      
      int kd = gap_size_v[j];

      prev_gap = H_left + P_W_OPEN;
      best_gap_h[i] += P_W_EXTEND;
      if (prev_gap > best_gap_h[i]) {
        best_gap_h[i] = prev_gap;
        gap_size_h[i] = 1;
      } else {
        gap_size_h[i]++;
      }

      int step_right = best_gap_h[i];
      step_right = (MATRIX_MIN_CUTOFF > step_right)? MATRIX_MIN_CUTOFF: step_right;

      int ki = gap_size_h[i];

      int bt_tmp;
      int sw_tmp = max(step_diag, step_down);
      sw_tmp = max(sw_tmp, step_right);
      bt_tmp = kd;
      bt_tmp = (sw_tmp == step_right)? -ki: bt_tmp;
      bt_tmp = (sw_tmp == step_diag)? 0: bt_tmp;

      H[i + 1][j + 1] = sw_tmp;
      bt[i][j] = bt_tmp;
    }
#pragma endscop
}
