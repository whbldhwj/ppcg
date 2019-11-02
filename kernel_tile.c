#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  /* PPCG generated CPU code */

#pragma scop
  for (int c0 = 0; c0 <= 511; c0 += 8)
    for (int c1 = 0; c1 <= 511; c1 += 8)
      for (int c2 = 0; c2 <= 511; c2 += 8)
        for (int c3 = 0; c3 <= 7; c3 += 1)
          for (int c4 = 0; c4 <= 7; c4 += 1) {
            if (c2 == 0)
              C[c0 + c3][c1 + c4] = 0;
            for (int c5 = 0; c5 <= 7; c5 += 1)
              C[c0 + c3][c1 + c4] = (C[c0 + c3][c1 + c4] + (A[c0 + c3][c2 + c5] * B[c2 + c5][c1 + c4]));
          }
#pragma endscop  
}
