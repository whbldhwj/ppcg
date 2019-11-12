/*
 * This code implements the convolutional neural network (point-wise convolution), which performs:
 * Z(o,r,c) += X(i,r+p-1,c+q-1) * W(o,i,p,q) 
 * Input: X[I][R+2][C+2], W[O][I][3][3]
 * Output: Z[O][I][R][C]
 */

#include "kernel.h"

int main(){
  // declarations
  data_t X[I][R + 2][C + 2];
  data_t W[O][I][3][3];
  data_t Z[O][R][C];
  data_t Z_dsa[O][R][C];

  // data initialization
  for (int i = 0 ; i < I; i++)
    for (int r = 0; r < R + 2; r++)
      for (int c = 0; c < C + 2; c++) {
        X[i][r][c] = (float)rand() / RAND_MAX;
      }

  for (int o = 0; o < O; o++)
    for (int i = 0; i < I; i++) 
      for (int p = 0; p < 3; p++)
        for (int q = 0; q < 3; q++) {
          W[o][i][p][q] = (float)rand() / RAND_MAX;
        }
  
  kernel(X, W, Z);
  hw_kernel(X, W, Z_dsa);

  // comparison
  int err = 0;
  float thres = 0.01;
  for (int o = 0; o < O; o++)
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++) {
        if (fabs(Z_dsa[o][r][c] - Z[o][r][c]) > thres) {
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
