#include "kernel.h"

/* DSA Form 0 */
// change parameters to constants
// avoid using +=
void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]) {
#pragma scop  
  for (int i = 0; i < 128; i++)
    for (int j = 0; j < 128; j++) 
      for (int k = 0; k < 128; k++) 
        for (int ii = 0; ii < 4; ii++)
          for (int jj = 0; jj < 4; jj++) 
            for (int kk = 0; kk < 4; kk++) {
              if (k * 4 + kk == 0)
                C[i * 4 + ii][j * 4 + jj] = 0;
              C[i * 4 + ii][j * 4 + jj] = C[i * 4 + ii][j * 4 + jj] 
                + A[i * 4 + ii][k * 4 + kk] * B[k * 4 + kk][j * 4 + jj];
            }
#pragma endscop
}
