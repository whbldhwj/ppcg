#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef float data_t;
#define I 64
#define J 64
#define K 64
#define L 64

void dsa_kernel(data_t A[I][J][L], data_t B[L][K], data_t C[I][J][K]);
