#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2, c3;

// Function declarations
#define FUNC_S0 type_of<float>(), {c0, c1, c2, c3}, Place::Host
#define FUNC_S1 type_of<float>(), {c0, c1, c2, c3}, Place::Host
#define FUNC_S2 type_of<float>(), {c0, c1, c2, c3}, Place::Host
#define FUNC_S3 type_of<float>(), {c0, c1, c2, c3}, Place::Host
#define FUNC_S4 type_of<float>(), {c0, c1, c2, c3}, Place::Host
Func A(FUNC_S0), B(FUNC_S1), C(FUNC_S2), D(FUNC_S3), D_drain(FUNC_S4);

// UREs
A(c0, c1, c2, c3) = 0;
A(c0, c1, c2, c3) = select(c1 == 0, A[c0][c2][c3], select(c1 + -1 >= 0, A(c0, c1 - 1, c2, c3), A(c0, c1, c2, c3)));
B(c0, c1, c2, c3) = 0;
B(c0, c1, c2, c3) = select(c0 == 0, B[c2][c1], select(c0 + -1 >= 0, B(c0 - 1, c1, c2, c3), B(c0, c1, c2, c3)));
C(c0, c1, c2, c3) = 0;
C(c0, c1, c2, c3) = select(c2 == 0, C[c3][c1], select(c2 + -1 >= 0, C(c0, c1, c2 - 1, c3), C(c0, c1, c2, c3)));
D(c0, c1, c2, c3) = 0;
D(c0, c1, c2, c3) = select((c3 == 0) && (c2 == 0), 0, D(c0, c1, c2, c3));
D(c0, c1, c2, c3 - 1) += select(c3 + -1 >= 0, ((A(c0, c1, c2, c3) * B(c0, c1, c2, c3)) * C(c0, c1, c2, c3)), D(c0, c1, c2, c3 - 1));
D(c0, c1, c2 - 1, c3 - -63) += select((c3 == 0) && (c2 + -1 >= 0), ((A(c0, c1, c2, c3) * B(c0, c1, c2, c3)) * C(c0, c1, c2, c3)), D(c0, c1, c2 - 1, c3 - -63));
D(c0, c1, c2, c3) += select((c3 == 0) && (c2 == 0), ((A(c0, c1, c2, c3) * B(c0, c1, c2, c3)) * C(c0, c1, c2, c3)), D(c0, c1, c2, c3));
D_drain(c0, c1, c2, c3) = 0;
D_drain(c0, c1, c2, c3) = select((c3 + -63 == 0) && (c2 + -63 == 0), D(c0, c1, c2, c3), D_drain(c0, c1, c2, c3));

// Space-time transformation
Var tloop0, tloop1, sloop0;
D_drain.merge_defs({A.update(0), B.update(0), C.update(0), D.update(0), D.update(1), D.update(2), D.update(3), D_drain.update(0)}, {A, B, C, D})
       .reorder_inward(c0, c1, c2, c3)
       .space_time_transform({c0, c1, c2, c3},
                             {sloop0},
                             {tloop0, tloop1},
                             {1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1},
                             {1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1})
       .domain(c0, 0, 63, 1,
               c1, 0, 63, 1,
               c2, 0, 63, 1,
               c3, 0, 63, 1,
               sloop0, 0, 63, 1,
               tloop0, 0, 63, 1,
               tloop1, 0, 63, 1,
               tloop2, 0, 63, 1);

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
