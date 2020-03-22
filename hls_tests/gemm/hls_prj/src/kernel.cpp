#include "kernel.h"
#include "kernel_kernel.h"

int main(int argc, char **argv) {
//  data_t A[I][K], B[K][J], C[I][J], C_golden[I][J]; 
  data_t A[I][K], B[J][K], C[I][J], C_golden[I][J];

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) {
      A[i][k] = k;
    }

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {
      B[j][k] = k;
    }

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C[i][j] = 0;
      C_golden[i][j] = 0;
    }

//#pragma scop
//  for (int i = 0; i < I; i++)
//    for (int j = 0; j < J; j++) {
//      C[i][j] = 0;
//      for (int k = 0; k < K; k++) {
//        C[i][j] = C[i][j] + A[i][k] * B[j][k];
//      }
//    }
//#pragma endscop
  kernel0((A_t4 *)A, (B_t4 *)B, (C_t2 *)C);
//  kernel0((data_t *)A, (data_t *)B, (data_t *)C);
//  kernel0((A_t2 *)A, (B_t2 *)B, (data_t *)C);

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      C_golden[i][j] = 0;
      for (int k = 0; k < K; k++) {
        C_golden[i][j] = C_golden[i][j] + A[i][k] * B[j][k];
      }
    }

  int err = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      if (abs(C_golden[i][j] - C[i][j]) > 0.001)
        err++;
    }

  if (err)
    printf("Failed with %d errors!\n", err);
  else
    printf("passed!\n");

  return 0;
}
