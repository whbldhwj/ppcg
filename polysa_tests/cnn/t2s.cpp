#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2, c3, c4, c5;

// Function declarations
#define FUNC_S0 type_of<float>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S1 type_of<float>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S2 type_of<float>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S3 type_of<float>(), {c0, c1, c2, c3, c4, c5}, Place::Host
Func W(FUNC_S0), X(FUNC_S1), Z(FUNC_S2), Z_drain(FUNC_S3);

// UREs
X(c0, c1, c2, c3, c4, c5) = 0;
X(c0, c1, c2, c3, c4, c5) = select(c0 == 0, X[c3][c1 + c4][c2 + c5], select(c0 + -1 >= 0, X(c0 - 1, c1, c2, c3, c4, c5), X(c0, c1, c2, c3, c4, c5)));
W(c0, c1, c2, c3, c4, c5) = 0;
W(c0, c1, c2, c3, c4, c5) = select(c2 == 0, W[c0][c3][c4][c5], select(c2 + -1 >= 0, W(c0, c1, c2 - 1, c3, c4, c5), W(c0, c1, c2, c3, c4, c5)));
Z(c0, c1, c2, c3, c4, c5) = 0;
Z(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c4 == 0) && (c3 == 0), 0, Z(c0, c1, c2, c3, c4, c5));
Z(c0, c1, c2, c3, c4, c5) = select(c5 + -1 >= 0, (Z(c0, c1, c2, c3, c4, c5 - 1) + (X(c0, c1, c2, c3, c4, c5) * W(c0, c1, c2, c3, c4, c5))), Z(c0, c1, c2, c3, c4, c5));
Z(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c4 + -1 >= 0), (Z(c0, c1, c2, c3, c4 - 1, c5 - -2) + (X(c0, c1, c2, c3, c4, c5) * W(c0, c1, c2, c3, c4, c5))), Z(c0, c1, c2, c3, c4, c5));
Z(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c4 == 0) && (c3 + -1 >= 0), (Z(c0, c1, c2, c3 - 1, c4 - -2, c5 - -2) + (X(c0, c1, c2, c3, c4, c5) * W(c0, c1, c2, c3, c4, c5))), Z(c0, c1, c2, c3, c4, c5));
Z(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c4 == 0) && (c3 == 0), (Z(c0, c1, c2, c3, c4, c5) + (X(c0, c1, c2, c3, c4, c5) * W(c0, c1, c2, c3, c4, c5))), Z(c0, c1, c2, c3, c4, c5));
Z_drain(c0, c1, c2, c3, c4, c5) = 0;
Z_drain(c0, c1, c2, c3, c4, c5) = select((c5 + -2 == 0) && (c4 + -2 == 0) && (c3 + -15 == 0), Z(c0, c1, c2, c3, c4, c5), Z_drain(c0, c1, c2, c3, c4, c5));

// Space-time transformation
Var tloop0, tloop1, tloop2, sloop0;
Z_drain.merge_defs({X.update(0), W.update(0), Z.update(0), Z.update(1), Z.update(2), Z.update(3), Z.update(4), Z_drain.update(0)}, {X, W, Z})
       .reorder_inward(c0, c1, c2, c3, c4, c5)
       .space_time_transform({c0, c1, c2, c3, c4, c5},
                             {sloop0},
                             {tloop0, tloop1, tloop2},
                             {1, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0,
                              0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 1},
                             {1, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0,
                              0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 1})
       .domain(c0, 0, 15, 1,
               c1, 0, 15, 1,
               c2, 0, 15, 1,
               c3, 0, 15, 1,
               c4, 0, 2, 1,
               c5, 0, 2, 1,
               sloop0, 0, 15, 1,
               tloop0, 0, 15, 1,
               tloop1, 0, 15, 1,
               tloop2, 0, 15, 1,
               tloop3, 0, 2, 1,
               tloop4, 0, 2, 1);

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
