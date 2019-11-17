#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2;

// Function declarations
#define FUNC_S0 type_of<int>(), {c0, c1, c2}, Place::Host
#define FUNC_S1 type_of<int>(), {c0, c1, c2}, Place::Host
#define FUNC_S2 type_of<int>(), {c0, c1, c2}, Place::Host
#define FUNC_S3 type_of<int>(), {c0, c1, c2}, Place::Host
Func A(FUNC_S0), B(FUNC_S1), C(FUNC_S2), C_drain(FUNC_S3);

// UREs
A(c0, c1, c2) = 0;
A(c0, c1, c2) = select(c1 == 0, A[c0][c2], select(c1 + -1 >= 0, A(c0, c1 - 1, c2), A(c0, c1, c2)));
B(c0, c1, c2) = 0;
B(c0, c1, c2) = select(c0 == 0, B[c2][c1], select(c0 + -1 >= 0, B(c0 - 1, c1, c2), B(c0, c1, c2)));
C(c0, c1, c2) = 0;
C(c0, c1, c2) = select(c2 == 0, 0, C(c0, c1, c2));
C(c0, c1, c2) = select(c2 == 0, (C(c0, c1, c2) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));
C(c0, c1, c2) = select(c2 + -1 >= 0, (C(c0, c1, c2 - 1) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));
C_drain(c0, c1, c2) = 0;
C_drain(c0, c1, c2) = select(c2 + -511 == 0, C(c0, c1, c2), C_drain(c0, c1, c2));

// Space-time transformation
Var tloop0, tloop1, sloop0;
C_drain.merge_defs({A.update(0), B.update(0), C.update(0), C.update(1), C.update(2), C_drain.update(0)}, {A, B, C})
       .reorder_inward(c0, c1, c2)
       .space_time_transform({c0, c1, c2},
                             {sloop0},
                             {tloop0, tloop1},
                             {1, 0, 0,
                              0, 1, 0,
                              0, 0, 1},
                             {1, 0, 0,
                              0, 1, 0,
                              0, 0, 1})
       .domain(c0, 0, 511, 1,
               c1, 0, 511, 1,
               c2, 0, 511, 1,
               sloop0, 0, 511, 1,
               tloop0, 0, 511, 1,
               tloop1, 0, 511, 1);

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
