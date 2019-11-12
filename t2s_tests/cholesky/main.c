/*
 * This code implements the Cholesky decomposition
 * Cholesky decomposition decompose a Hermitian positive-definite matrix A into one lower triangular matrix 
 * with real and positive diagonal entries L, and the conjugate transpose of L. 
 * Input: A[N][N]
 * Output: L1[N][N], L2[N][N]
 */

#include "kernel.h"

data_t print_mat(data_t* mat, int row, int col) {
  printf("****\n");
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%f\t", mat[i * col + j]);
    }
    printf("\n");
  }
  printf("****\n");
}

int main(){
  // declarations
  data_t A1[N][N];
  data_t A2[N][N];
  data_t L1[N][N];
  data_t L2[N][N];
  data_t L1_dsa[N][N];
  data_t L2_dsa[N][N];
  
  // data initialization
  // generate a positive-definite matrix as the input
  data_t T[N][N];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      if (j < i + 1) {
        T[i][j] = (float)rand() / RAND_MAX;
      } else {
        T[i][j] = 0;
      }
    }
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      A1[i][j] = 0;
      for (int k = 0; k < N; k++) {
        A1[i][j] += T[i][k] * T[j][k];
      }
    }

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
//      A1[i][j] = (float)rand() / RAND_MAX;
      A2[i][j] = A1[i][j];
    }

  // computation
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      L1[i][j] = 0;
    }
  }

  for (int i = 0; i < N; i++) {
    L1[i][i] = sqrtf(A1[i][i]);
    for (int j = i + 1; j < N; j++) {
      L1[j][i] = A1[j][i] / L1[i][i];
      for (int k = i + 1; k <= j; k ++) {
        A1[j][k] = A1[j][k] - L1[j][i] * L1[k][i];
      }
    }
  }

  dsa_kernel(A2, L1_dsa);

  print_mat((data_t *)T, N, N);
  print_mat((data_t *)A2, N, N);
  print_mat((data_t *)L1, N, N);
  print_mat((data_t *)L1_dsa, N, N);

  // comparison
  int err = 0;
  float thres = 0.001;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      if (fabs(L1_dsa[i][j] - L1[i][j]) > thres) {
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
