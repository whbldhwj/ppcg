#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef float data_t;
#define N 64
#define K 64

void dsa_kernel(data_t d[N][K], data_t V[K][K]);
