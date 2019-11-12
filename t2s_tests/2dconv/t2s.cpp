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
Func W(FUNC_S0), X(FUNC_S1), Z(FUNC_S2), Z_drain(FUNC_S3);

// UREs
X(c0, c1, c2, c3) = 0;
X(c0, c1, c2, c3) = select(((c2 == 0) && (-1 * c1 + 30 >= 0)) || (-1 * c1 + c2 + 31 == 0), X[c1][c0 + c3], select((-1 * c1 + c2 + 30 >= 0) && (c2 + -1 >= 0), X(c0, c1, c2 - 1, c3), X(c0, c1, c2, c3)));
W(c0, c1, c2, c3) = 0;
W(c0, c1, c2, c3) = select(-1 * c1 + c2 == 0, W[c2][c3], select(c1 + -1 * c2 + -1 >= 0, W(c0, c1 - 1, c2, c3), W(c0, c1, c2, c3)));
Z(c0, c1, c2, c3) = 0;
Z(c0, c1, c2, c3) = select((c3 == 0) && (c2 == 0), 0, Z(c0, c1, c2, c3));
Z(c0, c1, c2, c3) = select(c3 + -1 >= 0, (Z(c0, c1, c2, c3 - 1) + (X(c0, c1, c2, c3) * W(c0, c1, c2, c3))), Z(c0, c1, c2, c3));
Z(c0, c1, c2, c3) = select((c3 == 0) && (c2 + -1 >= 0), (Z(c0, c1 - 1, c2 - 1, c3 - -2) + (X(c0, c1, c2, c3) * W(c0, c1, c2, c3))), Z(c0, c1, c2, c3));
Z(c0, c1, c2, c3) = select((c3 == 0) && (c2 == 0), (Z(c0, c1, c2, c3) + (X(c0, c1, c2, c3) * W(c0, c1, c2, c3))), Z(c0, c1, c2, c3));
Z_drain(c0, c1, c2, c3) = 0;
Z_drain(c0, c1, c2, c3) = select((c3 + -2 == 0) && (c2 + -2 == 0), Z(c0, c1, c2, c3), Z_drain(c0, c1, c2, c3));

// Space-time transformation
Var tloop0, tloop1, sloop0;
Z_drain.merge_defs({X.update(0), W.update(0), Z.update(0), Z.update(1), Z.update(2), Z.update(3), Z_drain.update(0)}, {X, W, Z})
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
       .domain(c0, 0, 31, 1,
               c1, 0, 33, 1,
               c2, 0, 2, 1,
               c3, 0, 2, 1,
               sloop0, 0, 31, 1,
               tloop0, 0, 33, 1,
               tloop1, 0, 2, 1,
               tloop2, 0, 2, 1);

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
