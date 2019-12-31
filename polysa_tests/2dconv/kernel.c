#include "kernel.h"

void dsa_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]) {
#pragma scop
  for (int r = 0; r < R; r++)
    for (int s = 0; s < S; s++) {
//      Z[r][s] = 0;
      for (int i = 0; i < K; i++) 
        for (int j = 0; j < K; j++) {
          if (i == 0 && j == 0)
            Z[r][s] = 0;
          Z[r][s] = Z[r][s] + X[r + i][s + j] * W[i][j];
        }
    }
#pragma endscop
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
