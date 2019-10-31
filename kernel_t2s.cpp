#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;
void kernel0(int *A, int *B, int *C)
{

    for (int c0 = 0; c0 <= 511; c0 += 1)
      for (int c1 = 0; c1 <= 511; c1 += 1) {
        C[c0 * 512 + c1] = 0;
        for (int c2 = 0; c2 <= 511; c2 += 1)
          C[c0 * 512 + c1] = (C[c0 * 512 + c1] + (A[c0 * 512 + c2] * B[c2 * 512 + c1]));
      }
}
