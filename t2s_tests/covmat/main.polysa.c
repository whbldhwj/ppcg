/*
 * This code implements the calculation of variance-covariance matrix.
 * The detailed algorithm can be found at:
 * https://stattrek.com/matrix-algebra/covariance-matrix.aspx
 * d = X - 11'X(1/n)
 * V = d'd(1/n)
 * Input: X[N][K]
 * Output: V[K][K]
 */

#include "kernel.h"

int main(){
  // declarations
  data_t X[N][K];
  data_t d[N][K];
  data_t V[K][K];
  data_t V_dsa[K][K];

  // data initialization
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
      X[n][k] = (float)rand() / RAND_MAX;
    }

  // computation
