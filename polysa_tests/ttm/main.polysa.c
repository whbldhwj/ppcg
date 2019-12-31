/*
 * This code implements the Tensor Times Matrix (TTM), which performs:
 * C(i,j,k) += A(i,j,l) * B(l,k)
 * Input: A[I][J][L], B[L][K]
 * Output: C[I][J][K]
 */

#include "kernel.h"

int main(){
  // declarations
  static data_t A[I][J][L];
  static data_t B[L][K];
  static data_t C[I][J][K];
  static data_t C_dsa[I][J][K];

  // data initialization
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) 
      for (int l = 0; l < L; l++) {
        A[i][j][l] = (float)rand() / RAND_MAX;
      }
  for (int l = 0; l < L; l++)
    for (int k = 0; k < K; k++) {
      B[l][k] = (float)rand() / RAND_MAX;
    }

  // computation
  /* PPCG generated CPU code */
  
  for (int c0 = 0; c0 <= 63; c0 += 1)
    for (int c1 = 0; c1 <= 63; c1 += 1)
      for (int c2 = 0; c2 <= 63; c2 += 1) {
        C[c0][c1][c2] = 0;
        for (int c3 = 0; c3 <= 63; c3 += 1)
          C[c0][c1][c2] += (A[c0][c1][c3] * B[c3][c2]);
      }

  dsa_kernel(A, B, C_dsa);

  // comparison
  int err = 0;
  float thres = 0.001;
  for (int i = 0; i < I; i++) 
    for (int j = 0; j < J; j++) 
      for (int k = 0; k < K; k++) {
        if (fabs(C_dsa[i][j][k] - C[i][j][k]) > thres) {
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
