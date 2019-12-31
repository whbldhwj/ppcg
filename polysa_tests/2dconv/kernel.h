#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef float data_t;
#define R 32
#define S 32
#define K 3

void dsa_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]);
void hw_kernel(data_t X[R + K - 1][S + K - 1], data_t W[K][K], data_t Z[R][S]);
