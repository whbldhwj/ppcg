#include "kernel.h"

#ifndef PIVOT
void dsa_kernel(data_t A[N][N], data_t L[N][N], data_t U[N][N]) {
  data_t L_ext[N][N][N];
  data_t U_ext[N][N][N];
  data_t A_ext[N][N][N];
  data_t A_diag_ext[N][N][N];

#pragma scop
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) { // row 
      for (int k = i; k < N; k++) { // col
        // reuse at i-axis
        if (i == 0) {
          A_ext[i][j][k] = A[j][k];
        } else {
          A_ext[i][j][k] = A_ext[i - 1][j][k];
        }

        // reuse at j-axis
        if (j == i && k == i) {
          A_diag_ext[i][j][k] = A_ext[i][j][k];
        } else if (j > i && k == i) {
          A_diag_ext[i][j][k] = A_diag_ext[i][j - 1][k];
        }

        // reuse at j-axis
        if (j == i) {
          U[i][k] = A_ext[i][j][k];
          U_ext[i][j][k] = A_ext[i][j][k];
        } else {
          U_ext[i][j][k] = U_ext[i][j - 1][k]; 
        }

        // reuse at k-axis
        if (j > i && k == i) {
          L_ext[i][j][k] = A_ext[i][j][k] / A_diag_ext[i][j][k];
        } else if (j > i && k > i) {
          L_ext[i][j][k] = L_ext[i][j][k - 1];
        }

        if (j == i && k == i) {
          L[j][k] = 1;
        } else if (j > i && k == i) {
          L[j][k] = L_ext[i][j][k];
        }

        if (j > i && k > i) {
          A_ext[i][j][k] = A_ext[i][j][k] - L_ext[i][j][k] * U_ext[i][j][k];
        }
      }
    }
  }
#pragma endscop
}
#else
// w/ partial pivoting
#endif
