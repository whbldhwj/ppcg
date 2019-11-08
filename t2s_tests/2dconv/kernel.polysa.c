#include "kernel.h"

void dsa_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]) {
  /* PPCG generated CPU code */
  
  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
  for (int c0 = 0; c0 <= 31; c0 += 1)
    for (int c1 = 0; c1 <= 33; c1 += 1)
      for (int c2 = ppcg_max(0, c1 - 31); c2 <= ppcg_min(2, c1); c2 += 1)
        for (int c3 = 0; c3 <= 2; c3 += 1) {
          if (c0 == 0 && c2 == c1)
            Z[0][0] = 0;
          Z[c1 - c2][c0] = (Z[c1 - c2][c0] + (X[c1][c0 + c3] * W[c2][c3]));
        }
}
