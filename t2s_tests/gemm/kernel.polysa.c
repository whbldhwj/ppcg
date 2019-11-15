#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  /* PPCG generated CPU code */
  
  for (int c0 = 0; c0 <= 511; c0 += 1)
    for (int c1 = 0; c1 <= 511; c1 += 1)
      for (int c2 = 0; c2 <= 511; c2 += 1) {
        if (c1 == 0)
          C[c2][c0] = 0;
        C[c2][c0] = (C[c2][c0] + (A[c2][c1] * B[c1][c0]));
      }
}
