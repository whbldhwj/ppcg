#include "kernel.h"

void kernel(data_t X[I][R + 2][C + 2], data_t W[O][I][3][3], data_t Z[O][R][C]) {
  // computation
  /* PPCG generated CPU code */
  
  for (int c0 = 0; c0 <= 15; c0 += 1)
    for (int c1 = 0; c1 <= 15; c1 += 1)
      for (int c2 = 0; c2 <= 15; c2 += 1) {
        Z[c0][c1][c2] = 0;
        for (int c3 = 0; c3 <= 15; c3 += 1)
          for (int c4 = 0; c4 <= 2; c4 += 1)
            for (int c5 = 0; c5 <= 2; c5 += 1)
              Z[c0][c1][c2] = (Z[c0][c1][c2] + (X[c3][c1 + c4][c2 + c5] * W[c0][c3][c4][c5]));
      }
}

void hw_kernel(data_t X[I][R + 2][C + 2], data_t W[O][I][3][3], data_t Z[O][R][C]) {
  // computation
  for (int o = 0; o < O; o++)
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++) {
        Z[o][r][c] = 0;
        for (int i = 0; i < I; i++)
          for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) {
              Z[o][r][c] += X[i][r + p][c + q] * W[o][i][p][q];
            }
      }
}
