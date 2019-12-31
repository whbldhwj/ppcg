#include "kernel.h"

#ifndef PIVOT
void dsa_kernel(data_t A[N][N], data_t L[N][N], data_t U[N][N]) {
  data_t L_ext[N][N][N];
  data_t U_ext[N][N][N];
  data_t A_ext[N][N][N];
  data_t A_diag_ext[N][N][N];

  /* PPCG generated CPU code */
  
  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  {
    for (int c0 = 0; c0 <= 3; c0 += 1)
      for (int c1 = 0; c1 <= 3; c1 += 1)
        A_ext[0][c0][c1] = A[c0][c1];
    for (int c0 = 0; c0 <= 3; c0 += 1) {
      for (int c1 = 0; c1 < c0; c1 += 1)
        for (int c2 = c1; c2 <= 3; c2 += 1)
          U_ext[c1][c0][c2] = U_ext[c1][c0 - 1][c2];
      for (int c1 = 0; c1 < c0; c1 += 1)
        A_diag_ext[c1][c0][c1] = A_diag_ext[c1][c0 - 1][c1];
      if (c0 >= 1)
        for (int c1 = 0; c1 <= 3; c1 += 1) {
          for (int c2 = 0; c2 < ppcg_min(c0, c1); c2 += 1)
            L_ext[c2][c0][c1] = L_ext[c2][c0][c1 - 1];
          if (c1 >= 1)
            for (int c2 = 0; c2 <= ppcg_min(c0, c1); c2 += 1) {
              if (c2 >= 1)
                A_ext[c2][c0][c1] = A_ext[c2 - 1][c0][c1];
              if (c1 >= c2 + 1 && c0 >= c2 + 1)
                A_ext[c2][c0][c1] = (A_ext[c2][c0][c1] - (L_ext[c2][c0][c1] * U_ext[c2][c0][c1]));
            }
          if (c0 >= c1 + 1)
            L_ext[c1][c0][c1] = (A_ext[c1][c0][c1] / A_diag_ext[c1][c0][c1]);
        }
      for (int c1 = c0; c1 <= 3; c1 += 1)
        U_ext[c0][c0][c1] = A_ext[c0][c0][c1];
      A_diag_ext[c0][c0][c0] = A_ext[c0][c0][c0];
    }
    for (int c0 = 0; c0 <= 3; c0 += 1)
      for (int c1 = c0; c1 <= 3; c1 += 1)
        U[c0][c1] = A_ext[c0][c0][c1];
    for (int c0 = 1; c0 <= 3; c0 += 1)
      for (int c1 = 0; c1 < c0; c1 += 1)
        L[c0][c1] = L_ext[c1][c0][c1];
    for (int c0 = 0; c0 <= 3; c0 += 1)
      L[c0][c0] = 1;
  }
}
#else
// w/ partial pivoting
#endif
