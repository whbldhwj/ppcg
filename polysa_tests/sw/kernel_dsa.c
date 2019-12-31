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

#pragma scop
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      // reuse at j-axis
      if (j == 0) {
        alt_ext[i][j] = alt[i];
      } else {
        alt_ext[i][j] = alt_ext[i][j - 1];
      }

      // reuse at i-axis
      if (i == 0) {
        ref_ext[i][j] = ref[j];
      } else {
        ref_ext[i][j] = ref_ext[i - 1][j];
      }

      sim_score_ext[i][j] = (alt_ext[i][j] == ref_ext[i][j])? P_W_MATCH: P_W_MISMATCH;

      if (i == 0 || j == 0) {
        step_diag_ext[i][j] = (MATRIX_MIN_CUTOFF > sim_score_ext[i][j])? MATRIX_MIN_CUTOFF: sim_score_ext[i][j];
      } else {
        step_diag_ext[i][j] = (MATRIX_MIN_CUTOFF > (H_ext[i - 1][j - 1] + sim_score_ext[i][j]))? MATRIX_MIN_CUTOFF: (H_ext[i - 1][j - 1] + sim_score_ext[i][j]);
      }

      if (i == 0) {
        if (P_W_OPEN > P_W_EXTEND + LOW_INIT_VALUE) {
          best_gap_v_ext[i][j] = P_W_OPEN;
          gap_size_v_ext[i][j] = 1;
        } else {
          best_gap_v_ext[i][j] = P_W_EXTEND + LOW_INIT_VALUE;
          gap_size_v_ext[i][j]++;
        }
      } else {
        if (H_ext[i - 1][j] + P_W_OPEN > best_gap_v_ext[i - 1][j] + P_W_EXTEND) {
          best_gap_v_ext[i][j] = H_ext[i - 1][j] + P_W_OPEN;
          gap_size_v_ext[i][j] = 1;
        } else {
          best_gap_v_ext[i][j] = best_gap_v_ext[i - 1][j] + P_W_EXTEND;
          gap_size_v_ext[i][j] = gap_size_v_ext[i - 1][j] + 1;
        }
      }

      step_down_ext[i][j] = (MATRIX_MIN_CUTOFF > best_gap_v_ext[i][j])? MATRIX_MIN_CUTOFF: best_gap_v_ext[i][j];

      kd_ext[i][j] = gap_size_v_ext[i][j];

      if (j == 0) {
        if (P_W_OPEN > P_W_EXTEND + LOW_INIT_VALUE) {
          best_gap_h_ext[i][j] = P_W_OPEN;
          gap_size_h_ext[i][j] = 1;
        } else {
          best_gap_h_ext[i][j] = P_W_EXTEND + LOW_INIT_VALUE;
          gap_size_h_ext[i][j]++;
        } 
      } else {
        if (H_ext[i][j - 1] + P_W_OPEN > best_gap_h_ext[i][j - 1] + P_W_EXTEND) {
          best_gap_h_ext[i][j] = H_ext[i][j - 1] + P_W_OPEN;
          gap_size_h_ext[i][j] = 1;
        } else {
          best_gap_h_ext[i][j] = best_gap_h_ext[i][j - 1] + P_W_EXTEND;
          gap_size_h_ext[i][j] = gap_size_h_ext[i][j - 1] + 1;
        }
      }

      step_right_ext[i][j] = (MATRIX_MIN_CUTOFF > best_gap_h_ext[i][j])? MATRIX_MIN_CUTOFF: best_gap_h_ext[i][j];
        
      ki_ext[i][j] = gap_size_h_ext[i][j];

      sw_tmp1_ext[i][j] = max(step_diag_ext[i][j], step_down_ext[i][j]);
      sw_tmp2_ext[i][j] = max(sw_tmp1_ext[i][j], step_right_ext[i][j]);
      
      H_ext[i][j] = sw_tmp2_ext[i][j];
      H[i + 1][j + 1] = H_ext[i][j];

      bt_tmp1_ext[i][j] = (H_ext[i][j] == step_right_ext[i][j])? -ki_ext[i][j] : kd_ext[i][j];
      bt_tmp2_ext[i][j] = (H_ext[i][j] == step_diag_ext[i][j])? 0 : bt_tmp1_ext[i][j];

      bt[i][j] = bt_tmp2_ext[i][j];
    }
#pragma endscop
}
