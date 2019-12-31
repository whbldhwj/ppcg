#include "kernel.h"
#include "math.h"

int main(){
  data_t X[R + K - 1][S + K - 1];
  data_t W[K][K];
  data_t Z[R][S];
  data_t Z_hw[R][S];

  for (int r = 0; r < R + K - 1; r++)
    for (int s = 0; s < S + K - 1; s++) {
      X[r][s] = (data_t)rand() / RAND_MAX;
    }

  for (int i = 0; i < K; i++)
    for (int j = 0; j < K; j++) {
      W[i][j] = (data_t)rand() / RAND_MAX;
    }

  dsa_kernel(X, W, Z);
  hw_kernel(X, W, Z_hw);

  int err = 0;
  float thres = 0.001;
  for (int r = 0; r < R; r++) 
    for (int s = 0; s < S; s++) {
      if (fabs(Z_hw[r][s] - Z[r][s]) > thres)
        err++;
    }

  if (err) {
    printf("Test failed with %d errors!\n", err);
    return -1;
  } else {
    printf("Test passed!\n");
    return 0;
  }
}
