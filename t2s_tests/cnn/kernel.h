#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef float data_t;
#define O 16
#define I 16
#define R 16
#define C 16

void kernel(data_t X[I][R + 2][C + 2], data_t W[O][I][3][3], data_t Z[O][R][C]);
void hw_kernel(data_t X[I][R + 2][C + 2], data_t W[O][I][3][3], data_t Z[O][R][C]);
