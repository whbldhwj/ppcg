#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2, c3, c4;

// Function declarations
#define FUNC_S0 type_of<float>(), {c0, c1, c2, c3, c4}, Place::Host
#define FUNC_S1 type_of<float>(), {c0, c1, c2, c3, c4}, Place::Host
#define FUNC_S2 type_of<float>(), {c0, c1, c2, c3, c4}, Place::Host
#define FUNC_S3 type_of<float>(), {c0, c1, c2, c3, c4}, Place::Host
#define FUNC_S4 type_of<float>(), {c0, c1, c2, c3, c4}, Place::Host
Func A(FUNC_S0), B(FUNC_S1), C(FUNC_S2), D(FUNC_S3), D_drain(FUNC_S4);

// UREs
A(c0, c1, c2, c3, c4) = 0;
A(c0, c1, c2, c3, c4) = select(c1 == 0, A[c0][c3][c4], select(c1 + -1 >= 0, A(c0, c1 - 1, c2, c3, c4), A(c0, c1, c2, c3, c4)));
B(c0, c1, c2, c3, c4) = 0;
B(c0, c1, c2, c3, c4) = select(c2 == 0, B[c3][c1], select(c2 + -1 >= 0, B(c0, c1, c2 - 1, c3, c4), B(c0, c1, c2, c3, c4)));
C(c0, c1, c2, c3, c4) = 0;
C(c0, c1, c2, c3, c4) = select(c1 == 0, C[c4][c2], select(c1 + -1 >= 0, C(c0, c1 - 1, c2, c3, c4), C(c0, c1, c2, c3, c4)));
D(c0, c1, c2, c3, c4) = 0;
D(c0, c1, c2, c3, c4) = select((c4 == 0) && (c3 == 0), 0, D(c0, c1, c2, c3, c4));
D(c0, c1, c2, c3, c4) += select((c4 == 0) && (c3 == 0), ((A(c0, c1, c2, c3, c4) * B(c0, c1, c2, c3, c4)) * C(c0, c1, c2, c3, c4)), D(c0, c1, c2, c3, c4));
D(c0, c1, c2, c3, c4 - 1) += select(c4 + -1 >= 0, ((A(c0, c1, c2, c3, c4) * B(c0, c1, c2, c3, c4)) * C(c0, c1, c2, c3, c4)), D(c0, c1, c2, c3, c4 - 1));
D(c0, c1, c2, c3 - 1, c4 - -15) += select((c4 == 0) && (c3 + -1 >= 0), ((A(c0, c1, c2, c3, c4) * B(c0, c1, c2, c3, c4)) * C(c0, c1, c2, c3, c4)), D(c0, c1, c2, c3 - 1, c4 - -15));
D_drain(c0, c1, c2, c3, c4) = 0;
D_drain(c0, c1, c2, c3, c4) = select((c4 + -15 == 0) && (c3 + -15 == 0), D(c0, c1, c2, c3, c4), D_drain(c0, c1, c2, c3, c4));

// Space-time transformation
Var tloop0, tloop1, tloop2, sloop0;
D_drain.merge_defs({A.update(0), B.update(0), C.update(0), D.update(0), D.update(1), D.update(2), D.update(3), D_drain.update(0)}, {A, B, C, D})
       .reorder_inward(c0, c1, c2, c3, c4)
       .space_time_transform({c0, c1, c2, c3, c4},
                             {sloop0},
                             {tloop0, tloop1, tloop2},
                             {1, 0, 0, 0, 0,
                              0, 1, 0, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0,
                              0, 0, 0, 0, 1},
                             {1, 0, 0, 0, 0,
                              0, 1, 0, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0,
                              0, 0, 0, 0, 1})
       .domain(c0, 0, 15, 1,
               c1, 0, 15, 1,
               c2, 0, 15, 1,
               c3, 0, 15, 1,
               c4, 0, 15, 1,
               sloop0, 0, 15, 1,
               tloop0, 0, 15, 1,
               tloop1, 0, 15, 1,
               tloop2, 0, 15, 1,
               tloop3, 0, 15, 1);

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
