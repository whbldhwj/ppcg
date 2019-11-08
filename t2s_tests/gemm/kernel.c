#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
#pragma scop
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        if (k == 0)
          C[i][j] = 0;
        C[i][j] = C[i][j] + A[i][k] * B[k][j];
      }
    }
#pragma endscop
}
