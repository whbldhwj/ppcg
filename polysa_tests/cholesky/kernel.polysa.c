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
  /* PPCG generated CPU code */
  
  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  for (int c0 = 0; c0 <= 8; c0 += 1) {
    if (c0 % 2 == 0) {
      if (c0 >= 2) {
        if (c0 >= 4)
          x = (x - (A[(c0 / 2) - 1][(c0 / 2) - 2] * A[(c0 / 2) - 1][(c0 / 2) - 2]));
        p[(c0 / 2) - 1] = (1.0 / sqrt(x));
        if (c0 <= 6) {
          x = A[(c0 / 2) - 1][c0 / 2];
          if (c0 == 6)
            x = (x - (A[3][0] * A[2][0]));
        }
        for (int c1 = (c0 / 2) + 1; c1 <= ppcg_min(4, (c0 / 2) + 2); c1 += 1) {
          if (2 * c1 == c0 + 4) {
            if (c0 == 4) {
              A[3][1] = (x * p[1]);
            } else {
              x = A[0][3];
            }
          } else if (c0 >= 4) {
            for (int c2 = 2; c2 <= (-c0 / 2) + 5; c2 += 1) {
              x = (x - (A[(c0 / 2) + c2 - 2][(c0 / 2) - 2] * A[(c0 / 2) - 1][(c0 / 2) - 2]));
              if (c2 == 2) {
                A[c0 / 2][(c0 / 2) - 1] = (x * p[(c0 / 2) - 1]);
                if (c0 == 4)
                  x = A[1][3];
              }
            }
          } else {
            x = A[0][2];
          }
          if (c1 == 4) {
            x = A[c0 / 2][c0 / 2];
            if (c0 == 6)
              x = (x - (A[3][0] * A[3][0]));
          }
        }
      }
      if (c0 <= 2)
        x = A[c0 / 2][c0 / 2];
    } else if (c0 >= 5) {
      x = (x - (A[(c0 - 1) / 2][(c0 - 5) / 2] * A[(c0 - 1) / 2][(c0 - 5) / 2]));
    }
  }
}
