/*
 * This code implements the Smith-Waterman algorithm
 * The implmentation is extracted from HaplotypeCaller from GATK pipeline.
 * There are two steps in SW:
 * - the construction of the scoring matrix H and the back-tracing matrix bt
 * - the construction of the optimal alignment based on the last row and column of H and
 *   the back-tracing matrix bt
 * Note that in this code, we only implement the first step. 
 * Input: alt[M], ref[N]
 * Output: H[M+1][N+1], bt[M][N]
 */

#include "kernel.h"

void print_mat(data_t* mat, int row, int col) {
  printf("****\n");
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%d\t", mat[i * col + j]);
    }
    printf("\n");
  }
  printf("****\n");
}

int main(){
  // declarations
  char alt[M];
  char ref[N];
  int bt[M][N];
  int H[M + 1][N + 1];
  int best_gap_h[M];
  int gap_size_h[M];
  int best_gap_v[N];
  int gap_size_v[N];

  int bt_dsa[M][N];
  int H_dsa[M + 1][N + 1];

  // data initialization
  for (int i = 0; i < M; i++)
    alt[i] = rand();
  for (int i = 0; i < N; i++)
    ref[i] = rand();
  
  // computation
  // 1. initialization
  for (int i = 0; i < M; i++)
    best_gap_h[i] = LOW_INIT_VALUE;
  for (int j = 0; j < N; j++)
    best_gap_v[j] = LOW_INIT_VALUE;
  for (int i = 0; i < M; i++)
    H[i + 1][0] = 0;
  for (int j = 0; j < N; j++)
    H[0][j + 1] = 0;
  H[0][0] = 0;
  for (int i = 0; i < M; i++)
    gap_size_h[i] = 0;
  for (int j = 0; j < N; j++)
    gap_size_v[j] = 0;  

  for (int i = 0; i < M; i++)
    H_dsa[i + 1][0] = 0;
  for (int j = 0; j < N; j++)
    H_dsa[0][j + 1] = 0;
  H_dsa[0][0] = 0;

  // 2. construction of H and BT
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

  dsa_kernel(alt, ref, H_dsa, bt_dsa);

  print_mat((int *)H, M + 1, N + 1);
  print_mat((int *)H_dsa, M + 1, N + 1);
  print_mat((int *)bt, M, N);
  print_mat((int *)bt_dsa, M, N);

  // comparison
  int err = 0;
  float thres = 0.001;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      if (abs(H_dsa[i + 1][j + 1] - H[i + 1][j + 1]) > thres) {
        err++;
      }
    }

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      if (abs(bt_dsa[i][j] - bt[i][j]) > thres) {
        err++;
      }
    }

  if (err) {
    printf("Test failed with %d errors!\n", err);
    return -1;
  } else {
    printf("Test passed!\n");
    return 0;
  }
}
