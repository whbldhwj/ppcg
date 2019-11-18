#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  /* PPCG generated CPU code */
#pragma scop  
  for (int c0 = 0; c0 <= 63; c0 += 1)
    for (int c1 = 0; c1 <= 63; c1 += 1)
      for (int c2 = 0; c2 <= 63; c2 += 1) {
        // array
        for (int c3 = 0; c3 <= 7; c3 += 1)
          for (int c4 = 0; c4 <= 7; c4 += 1) {
            if (c2 == 0)
              C[8 * c0 + c3][8 * c1 + c4] = 0;
            for (int c5 = 0; c5 <= 7; c5 += 1)
              C[8 * c0 + c3][8 * c1 + c4] = (C[8 * c0 + c3][8 * c1 + c4] + (A[8 * c0 + c3][8 * c2 + c5] * B[8 * c2 + c5][8 * c1 + c4]));
          }
      }
#pragma endscop  
}
