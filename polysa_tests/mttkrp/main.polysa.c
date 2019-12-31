/*
 * This code implements the Matricized Tensor Times Khatri-Rao Product (MTTKRP), which performs:
 * D(i,j) += A(i,k,l) * B(k,j) * C(l,j)
 * Input: A[I][K][L], B[K][J], C[L][J]
 * Output: D[I][J]
 */

#include "kernel.h"

int main(){
  // declarations
  static data_t A[I][K][L];
  static data_t B[K][J];
  static data_t C[L][J];
  static data_t D[I][J];
  static data_t D_dsa[I][J];

  // data initialization
  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) 
      for (int l = 0; l < L; l++) {
        A[i][k][l] = (float)rand() / RAND_MAX;
      }
  for (int k = 0; k < K; k++)
    for (int j = 0; j < J; j++) {
      B[k][j] = (float)rand() / RAND_MAX;
    }
  for (int l = 0; l < L; l++)
    for (int j = 0; j < J; j++) {
      C[l][j] = (float)rand() / RAND_MAX;
    }

  // computation
  /* PPCG generated CPU code */
  
  for (int c0 = 0; c0 <= 63; c0 += 1)
    for (int c1 = 0; c1 <= 63; c1 += 1) {
      D[c0][c1] = 0;
      for (int c2 = 0; c2 <= 63; c2 += 1)
        for (int c3 = 0; c3 <= 63; c3 += 1)
          D[c0][c1] += ((A[c0][c2][c3] * B[c2][c1]) * C[c3][c1]);
    }

  dsa_kernel(A, B, C, D_dsa);

  // comparison
  int err = 0;
  float thres = 0.001;
  for (int i = 0; i < I; i++) 
    for (int j = 0; j < J; j++) {
      if (fabs(D_dsa[i][j] - D[i][j]) > thres) {
        err++;
      }
    }

  if (err) {
    printf("Test failed with %d errors!\n", err);
    return -1;
  } else {
    printf("Test passed!\n");
    return 0;
  }
}
