#include "Halide.h"
#include <iostream>

using namespace Halide;
using namespace std;


int main(void) {
// Inputs (Fill in manually)

// Variable declarations
Var c0, c1, c2, c3, c4, c5, c6, c7;

// Function declarations
#define FUNC_S0 type_of<int>(), {c0, c1, c2, c3, c4, c5, c6, c7}, Place::Host
#define FUNC_S1 type_of<int>(), {c0, c1, c2, c3, c4, c5, c6, c7}, Place::Host
#define FUNC_S2 type_of<int>(), {c0, c1, c2, c3, c4, c5, c6, c7}, Place::Host
#define FUNC_S3 type_of<int>(), {c0, c1, c2, c3, c4, c5, c6, c7}, Place::Host
Func A(FUNC_S0), B(FUNC_S1), C(FUNC_S2), C_drain(FUNC_S3);

// UREs
A(c0, c1, c2, c3, c4, c5, c6, c7) = 0;
A(c0, c1, c2, c3, c4, c5, c6, c7) = select(c7 == 0, A[8c1 + c5 + 4c6][8c0 + c3], select(c7 + -1 == 0, A(c0, c1, c2, c3, c4, c5, c6, c7 - 1), A(c0, c1, c2, c3, c4, c5, c6, c7)));
B(c0, c1, c2, c3, c4, c5, c6, c7) = 0;
B(c0, c1, c2, c3, c4, c5, c6, c7) = select(c6 == 0, B[8c0 + c3][8c2 + c4 + 4c7], select(c6 + -1 == 0, B(c0, c1, c2, c3, c4, c5, c6 - 1, c7), B(c0, c1, c2, c3, c4, c5, c6, c7)));
C(c0, c1, c2, c3, c4, c5, c6, c7) = 0;
C(c0, c1, c2, c3, c4, c5, c6, c7) = select((c3 == 0) && (c0 == 0), 0, C(c0, c1, c2, c3, c4, c5, c6, c7));
C(c0, c1, c2, c3, c4, c5, c6, c7) = select(c3 + -1 >= 0, (C(c0, c1, c2, c3 - 1, c4, c5, c6, c7) + (A(c0, c1, c2, c3, c4, c5, c6, c7) * B(c0, c1, c2, c3, c4, c5, c6, c7))), C(c0, c1, c2, c3, c4, c5, c6, c7));
C(c0, c1, c2, c3, c4, c5, c6, c7) = select((c3 == 0) && (c0 + -1 >= 0), (C(c0 - 1, c1, c2, c3 - -7, c4, c5, c6, c7) + (A(c0, c1, c2, c3, c4, c5, c6, c7) * B(c0, c1, c2, c3, c4, c5, c6, c7))), C(c0, c1, c2, c3, c4, c5, c6, c7));
C(c0, c1, c2, c3, c4, c5, c6, c7) = select((c3 == 0) && (c0 == 0), (C(c0, c1, c2, c3, c4, c5, c6, c7) + (A(c0, c1, c2, c3, c4, c5, c6, c7) * B(c0, c1, c2, c3, c4, c5, c6, c7))), C(c0, c1, c2, c3, c4, c5, c6, c7));
C_drain(c0, c1, c2, c3, c4, c5, c6, c7) = 0;
C_drain(c0, c1, c2, c3, c4, c5, c6, c7) = select((c3 + -7 == 0) && (c0 + -63 == 0), C(c0, c1, c2, c3, c4, c5, c6, c7), C_drain(c0, c1, c2, c3, c4, c5, c6, c7));

// Space-time transformation (Fill in manually)

// PE optimization (Fill in manually)

// CPU verification (Fill in manually)

}
