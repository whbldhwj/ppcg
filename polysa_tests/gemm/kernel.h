#include "stdio.h"
#include "stdlib.h"
#include "math.h"

typedef int data_t;
#define I 512 
#define J 512
#define K 512

void dsa_kernel(data_t A[I][K], data_t B[K][J], data_t C[I][J]);
