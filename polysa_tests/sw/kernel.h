#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef int data_t;
#define M 16
#define N 8

#define P_W_MATCH 200
#define P_W_MISMATCH -150
#define P_W_OPEN -260
#define P_W_EXTEND -11
#define MAX_STATE_NUM 1552
#define MATRIX_MIN_CUTOFF -100000000
#define LOW_INIT_VALUE -1073741824
#define MAX_CORE_DATA -2147483648

#define max(a,b) ((a>b)?a:b)

void dsa_kernel(char alt[M], char ref[N], int H[M + 1][N + 1], int bt[M][N]);
