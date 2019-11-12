#include "kernel.h"

void dsa_kernel(data_t A[N][N], data_t L[N][N]) {
  // Distribute A
  // Compute and pass L1[i][i]
  // Compute and pass L1[j][i]
  // Compute and pass L1[k][i]
  
//  data_t A_ext[N][N][N];
//  data_t L1_diag_ext[N][N][N];
//  data_t L1_h_ext[N][N][N];
//  data_t L1_v_ext[N][N][N];
//
//#pragma scop
//  for (int i = 0; i < N; i++) {
//    for (int j = i ; j < N; j++) {
//      for (int k = i; k <= j; k++) {
//        if (i == 0) {
//          if (!(j > i && k > i)) {
//            A_ext[i][j][k] = A[j][k];
//          }
//        } else {
//          if (!(j > i && k > i)) {
//            A_ext[i][j][k] = A_ext[i - 1][j][k];
//          }
//        }
//
//        if (j == i && k == i) {
//          L1_diag_ext[i][j][k] = sqrtf(A_ext[i][j][k]);
//          L[j][k] = L1_diag_ext[i][j][k];
//        } else if (j > i && k == i) {
//          L1_diag_ext[i][j][k] = L1_diag_ext[i][j - 1][k];
//        }
//
//        if (j > i && k == i) {
//          L1_h_ext[i][j][k] = A_ext[i][j][k] / L1_diag_ext[i][j][k];
//          L[j][k] = L1_h_ext[i][j][k];
//        } else {
//          L1_h_ext[i][j][k] = L1_h_ext[i][j][k - 1];
//        }
//
//        if (j == k) {
//          L1_v_ext[i][j][k] = L1_h_ext[i][j][k];
//        } else {
//          L1_v_ext[i][j][k] = L1_v_ext[i][j - 1][k];
//        }
//
//        if (i == 0) {
//          if (j > i && k > i) {
//            A_ext[i][j][k] = A[j][k] - L1_h_ext[i][j][k] * L1_v_ext[i][j][k];
//          }
//        } else {
//          if (j > i && k > i) {
//            A_ext[i][j][k] = A_ext[i - 1][j][k] - L1_h_ext[i][j][k] * L1_v_ext[i][j][k];
//          }
//        }
//      }
//    }
//  }
//#pragma endscop

//#pragma scop
//  for (int i = 0; i < N; i++) {
//    L[i][i] = sqrtf(A[i][i]);
//    for (int j = i + 1; j < N; j++) {
//      L[j][i] = A[j][i] / L[i][i];
//      for (int k = i + 1; k <= j; k ++) {
//        A[j][k] = A[j][k] - L[j][i] * L[k][i];
//      }
//    }
//  }
//#pragma endscop

//#pragma scop
//  for (int i = 0; i < N; i++) {
//    for (int j = 0; j < i - 1; j++)  {
//      for (int k = 0; k < j - 1; k++) 
//        A[i][j] = A[i][j] - A[i][k] * A[j][k];
//      A[i][j] = A[i][j] / A[j][j];
//    }
//    for (int m = 0; m < i - 1; m++)
//      A[i][i] = A[i][i] - A[i][m] * A[i][m];
//    A[i][i] = sqrtf(A[i][i]);
//  }
//#pragma endscop
}

void new_kernel(data_t p[N], data_t A[N][N]) {
  data_t x;
#pragma scop
  for (int i = 0; i < N; i++)
  {
    x = A[i][i];
    for (int j = 0; j <= i - 1; j++)
      x = x - A[i][j] * A[i][j];
    p[i] = 1.0 / sqrt(x);
    for (int j = i + 1; j < N; j++) {
      x = A[i][j];
      for (int k = 0; k <= i - 1; k++) {
        x = x - A[j][k] * A[i][k];
        A[j][i] = x * p[i];
      }
    }
  }
#pragma endscop
}
