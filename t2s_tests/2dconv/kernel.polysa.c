#include "kernel.h"

void dsa_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]) {
  /* PPCG generated CPU code */
  
  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
  for (int c0 = 0; c0 <= 31; c0 += 1)
    for (int c1 = 0; c1 <= 33; c1 += 1) {
      if (c1 <= 31)
        Z[c1][c0] = 0;
      for (int c2 = ppcg_max(0, c1 - 31); c2 <= ppcg_min(2, c1); c2 += 1)
        for (int c3 = 0; c3 <= 2; c3 += 1)
          Z[c1 - c2][c0] = (Z[c1 - c2][c0] + (X[c1][c0 + c3] * W[c2][c3]));
    }
}

void hw_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]) {
  data_t X_ext[32][34][3][3];
  data_t W_ext[32][34][3][3];
  data_t Z_ext[32][34][3][3];
  for (int c0 = 0; c0 <= 31; c0++)
    for (int c1 = 0; c1 <= 33; c1++)
      for (int c2 = 0; c2 <= 2; c2++)
        for (int c3 = 0; c3 <= 2; c3++) {
//          printf("%d %d %d %d\n", c0, c1, c2, c3);
//          if (c0 == 24 && c1 == 1 && c2 == 2 && c3 == 1) {
//            printf("here\n");
//          }
          if (((c2 == 0) && (-1 * c1 + 30 >= 0)) || (-1 * c1 + c2 + 31 == 0)) {
            X_ext[c0][c1][c2][c3] = X[c1][c0 + c3];
          } else if ((-1 * c1 + c2 + 30 >= 0) && (c2 + -1 >= 0)) {
            X_ext[c0][c1][c2][c3] = X_ext[c0][c1][c2 - 1][c3];
          }

          if (-1 * c1 + c2 == 0)
            W_ext[c0][c1][c2][c3] = W[c2][c3];
          else if (c1 + -1 * c2 + -1 >= 0)
            W_ext[c0][c1][c2][c3] = W_ext[c0][c1 - 1][c2][c3];

          if ((c3 == 0) && (c2 == 0)) 
            Z_ext[c0][c1][c2][c3] = 0;
          else 
            Z_ext[c0][c1][c2][c3] = Z_ext[c0][c1][c2][c3];

          if (c3 + -1 >= 0)
            Z_ext[c0][c1][c2][c3] = (Z_ext[c0][c1][c2][c3 - 1] + (X_ext[c0][c1][c2][c3] * W_ext[c0][c1][c2][c3]));
          else
            Z_ext[c0][c1][c2][c3] = Z_ext[c0][c1][c2][c3];

          if ((c3 == 0) && (c2 + -1 >= 0))
//            Z_ext[c0][c1][c2][c3] = (Z_ext[c0][c1 - 1][c2 - 1][c3] + (X_ext[c0][c1][c2][c3] * W_ext[c0][c1][c2][c3]));
            Z_ext[c0][c1][c2][c3] = (Z_ext[c0][c1 - 1][c2 - 1][c3 + 2] + (X_ext[c0][c1][c2][c3] * W_ext[c0][c1][c2][c3]));
          else
            Z_ext[c0][c1][c2][c3] = Z_ext[c0][c1][c2][c3];

          if ((c3 == 0) && (c2 == 0))
//            Z_ext[c0][c1][c2][c3] = (Z_ext[c0][c1][c2][c3 - 1] + (X_ext[c0][c1][c2][c3] * W_ext[c0][c1][c2][c3]));
            Z_ext[c0][c1][c2][c3] = (Z_ext[c0][c1][c2][c3] + (X_ext[c0][c1][c2][c3] * W_ext[c0][c1][c2][c3]));
          else
            Z_ext[c0][c1][c2][c3] = Z_ext[c0][c1][c2][c3];

          if ((c3 + -2 == 0) && (c2 + -2 == 0) && (c1 - c2 >= 0) && (c0 >= 0))
            Z[c1 - c2][c0] = Z_ext[c0][c1][c2][c3];
        }
}
