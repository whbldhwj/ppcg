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
