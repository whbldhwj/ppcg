#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2, c3, c4, c5;

// Function declarations
#define FUNC_S0 type_of<int>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S1 type_of<int>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S2 type_of<int>(), {c0, c1, c2, c3, c4, c5}, Place::Host
#define FUNC_S3 type_of<int>(), {c0, c1, c2, c3, c4, c5}, Place::Host
Func A(FUNC_S0), B(FUNC_S1), C(FUNC_S2), C_drain(FUNC_S3);

// UREs
A(c0, c1, c2, c3, c4, c5) = 0;
A(c0, c1, c2, c3, c4, c5) = select(c4 == 0, A[8c0 + c3][8c2 + c5], select(c4 + -1 >= 0, A(c0, c1, c2, c3, c4 - 1, c5), A(c0, c1, c2, c3, c4, c5)));
B(c0, c1, c2, c3, c4, c5) = 0;
B(c0, c1, c2, c3, c4, c5) = select(c3 == 0, B[8c2 + c5][8c1 + c4], select(c3 + -1 >= 0, B(c0, c1, c2, c3 - 1, c4, c5), B(c0, c1, c2, c3, c4, c5)));
C(c0, c1, c2, c3, c4, c5) = 0;
C(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c2 == 0), 0, C(c0, c1, c2, c3, c4, c5));
C(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c2 == 0), (C(c0, c1, c2, c3, c4, c5) + (A(c0, c1, c2, c3, c4, c5) * B(c0, c1, c2, c3, c4, c5))), C(c0, c1, c2, c3, c4, c5));
C(c0, c1, c2, c3, c4, c5) = select(c5 + -1 >= 0, (C(c0, c1, c2, c3, c4, c5 - 1) + (A(c0, c1, c2, c3, c4, c5) * B(c0, c1, c2, c3, c4, c5))), C(c0, c1, c2, c3, c4, c5));
C(c0, c1, c2, c3, c4, c5) = select((c5 == 0) && (c2 + -1 >= 0), (C(c0, c1, c2 - 1, c3, c4, c5 - -7) + (A(c0, c1, c2, c3, c4, c5) * B(c0, c1, c2, c3, c4, c5))), C(c0, c1, c2, c3, c4, c5));
C_drain(c0, c1, c2, c3, c4, c5) = 0;
C_drain(c0, c1, c2, c3, c4, c5) = select((c5 + -7 == 0) && (c2 + -63 == 0), C(c0, c1, c2, c3, c4, c5), C_drain(c0, c1, c2, c3, c4, c5));

// Space-time transformation (Fill in manually)

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
