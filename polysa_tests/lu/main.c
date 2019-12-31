/*
 * This code implements the LU decomposition using Doolittle method
 * LU decompostion decomposes the matrix A into one lower triangular matrix L and one upper triangular matrix U.
 * Input: A[N][N]
 * Output: L[N][N], U[N][N]
 */

#include "kernel.h"

// compute the magnitude of (a, b)
// mag = sqrt(a^2+b^2)
data_t compute_mag(data_t a, data_t b) {
  data_t aa = a * a;
  data_t bb = b * b;
  data_t mag = sqrtf(aa + bb);
  return mag;
}

data_t compute_mm(data_t c, data_t s, data_t *op1, data_t *op2) {
  data_t a = *op1 * c + *op2 * s;
  data_t b = -(*op1) * s + *op2 * c;

  *op1 = a;
  *op2 = b;
}

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
  data_t L[N][N];
  data_t U[N][N];
  data_t P[N];
  data_t L_dsa[N][N];
  data_t U_dsa[N][N];

#ifdef PIVOT
  printf("Partial pivoting enabled.\n");
#else
  printf("Partial pivoting disabled.\n");
#endif

  // data initialization
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      A1[i][j] = (float)rand() / RAND_MAX;
      A2[i][j] = A1[i][j];
    }

  // computation
  // 1. initialization
  for (int i = 0; i < N; i++) {
    P[i] = 0;
    for (int j = 0; j < N; j++) {      
      U[i][j] = 0;
      if (i == j)
        L[i][j] = 1;
      else 
        L[i][j] = 0;
    }
  }

  // 2. apply Doolittle's method
  for (int i = 0; i < N - 1; i++) {
    data_t maxpwr = fabs(A1[i][i]);
#ifdef PIVOT    
    P[i] = i;
    for (int j = i + 1; j < N; j++) {
      if (maxpwr < abs(A1[j][i])) {
        maxpwr = fabs(A1[j][i]);
        P[i] = j;
      }
    }
    if (P[i] != i) {
      int index = P[i];
      for (int j = i; j < N; j++) {
        maxpwr = A1[index][j];
        A1[index][j] = A1[i][j];
        A1[i][j] = maxpwr;
      }
    }
#endif    

    for (int j = i; j < N; j++)
      U[i][j] = A1[i][j];
    for (int j = i + 1; j < N; j++)
      L[j][i] = A1[j][i] / A1[i][i];
    for (int j = i + 1; j < N; j++) {
      for (int k = i + 1; k < N; k++) {
        A1[k][j] = A1[k][j] - L[k][i] * U[i][j];
      }
    }
  }
  P[N - 1] = N - 1;
  U[N - 1][N - 1] = A1[N - 1][N - 1];

  dsa_kernel(A2, L_dsa, U_dsa);

  print_mat((data_t *)A2, N, N);
  print_mat((data_t *)L, N, N);
  print_mat((data_t *)U, N, N);
  print_mat((data_t *)L_dsa, N, N);
  print_mat((data_t *)U_dsa, N, N);

  // comparison
  int err = 0;
  float thres = 0.001;
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) {
      if (fabs(L_dsa[i][j] - L[i][j]) > thres) {
        err++;
      }
    }

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      if (fabs(U_dsa[i][j] - U[i][j]) > thres) {
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
