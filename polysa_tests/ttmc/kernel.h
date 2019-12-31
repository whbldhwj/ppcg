#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef float data_t;
#define I 16 
#define J 16 
#define K 16 
#define L 16 
#define M 16 

void dsa_kernel(data_t A[I][L][M], data_t B[L][J], data_t C[M][K], data_t D[I][J][K]);
