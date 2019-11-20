#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
  /* PPCG generated CPU code */
  
  for (int c0 = 0; c0 <= 63; c0 += 1)
    for (int c1 = 0; c1 <= 63; c1 += 1)
      for (int c2 = 0; c2 <= 63; c2 += 1) {
        // array
        for (int c3 = 0; c3 <= 7; c3 += 1) {
          // latency
          for (int c4 = 0; c4 <= 3; c4 += 1) {
            // latency
            for (int c5 = 0; c5 <= 3; c5 += 1)
              for (int c6 = 0; c6 <= 1; c6 += 1)
                for (int c7 = 0; c7 <= 1; c7 += 1) {
                  if (c0 == 0 && c3 == 0)
                    C[8 * c1 + c5 + 4 * c6][8 * c2 + c4 + 4 * c7] = 0;
                  C[8 * c1 + c5 + 4 * c6][8 * c2 + c4 + 4 * c7] = (C[8 * c1 + c5 + 4 * c6][8 * c2 + c4 + 4 * c7] + (A[8 * c1 + c5 + 4 * c6][8 * c0 + c3] * B[8 * c0 + c3][8 * c2 + c4 + 4 * c7]));
                }
          }
        }
      }
}
