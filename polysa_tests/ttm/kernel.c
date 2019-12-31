#include "kernel.h"

void dsa_kernel(data_t A[I][J][L], data_t B[L][K], data_t C[I][J][K]){
  static data_t A_ext[I][J][K][L];
  static data_t B_ext[I][J][K][L];
  static data_t C_ext[I][J][K][L];

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        for (int l = 0; l < L; l++) {
          if (k == 0) {
            A_ext[i][j][k][l] = A[i][j][l];
          } else {
            A_ext[i][j][k][l] = A_ext[i][j][k - 1][l];
          }
          
          // reuse at i-axis
          if (i == 0) {
            if (j == 0) {
              B_ext[i][j][k][l] = B[l][k];
            } else {
              // initial distribution at j-axis
              B_ext[i][j][k][l] = B_ext[i][j - 1][k][l];
            }
          } else {
            B_ext[i][j][k][l] = B_ext[i - 1][j][k][l];
          }

          if (l == 0) {
            C_ext[i][j][k][l] = A_ext[i][j][k][l] * B_ext[i][j][k][l];
          } else {            
            C_ext[i][j][k][l] = C_ext[i][j][k][l - 1] + A_ext[i][j][k][l] * B_ext[i][j][k][l];
          }
        }     
        C[i][j][k] = C_ext[i][j][k][L - 1];
      }
    }
  }
}
