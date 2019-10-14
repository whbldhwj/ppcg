#include <assert.h>
#include <stdio.h>
#include "kernel_kernel.hu"
#include "kernel.h"

void dsa_kernel(char alt[M], char ref[N], int H[M + 1][N + 1], int bt[M][N]) {
//  char alt_ext[M][N];
//  char ref_ext[M][N];
  int sim_score_ext[M][N];
  int H_ext[M][N];

  int step_diag_ext[M][N];
 
  int best_gap_v_ext[M][N];
  int gap_size_v_ext[M][N];
  int step_down_ext[M][N];
  int kd_ext[M][N];
  
  int best_gap_h_ext[M][N];
  int gap_size_h_ext[M][N];
  int step_right_ext[M][N];
  int ki_ext[M][N];

  int sw_tmp1_ext[M][N];
  int sw_tmp2_ext[M][N];
  int bt_tmp1_ext[M][N];
  int bt_tmp2_ext[M][N];

  {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

    int *dev_H;
    int *dev_H_ext;
    char *dev_alt;
    int *dev_best_gap_h_ext;
    int *dev_best_gap_v_ext;
    int *dev_bt;
    int *dev_gap_size_h_ext;
    int *dev_gap_size_v_ext;
    char *dev_ref;
    int *dev_sim_score_ext;
    int *dev_step_diag_ext;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_H, (17) * (9) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_H_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_alt, (16) * sizeof(char)));
    cudaCheckReturn(cudaMalloc((void **) &dev_best_gap_h_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_best_gap_v_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_bt, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_gap_size_h_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_gap_size_v_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_ref, (8) * sizeof(char)));
    cudaCheckReturn(cudaMalloc((void **) &dev_sim_score_ext, (16) * (8) * sizeof(int)));
    cudaCheckReturn(cudaMalloc((void **) &dev_step_diag_ext, (16) * (8) * sizeof(int)));
    
    cudaCheckReturn(cudaMemcpy(dev_H, H, (17) * (9) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_alt, alt, (16) * sizeof(char), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_best_gap_h_ext, best_gap_h_ext, (16) * (8) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_best_gap_v_ext, best_gap_v_ext, (16) * (8) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_gap_size_h_ext, gap_size_h_ext, (16) * (8) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_gap_size_v_ext, gap_size_v_ext, (16) * (8) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_ref, ref, (8) * sizeof(char), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(8, 16);
      dim3 k0_dimGrid(1, 1);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_alt, dev_ref, dev_sim_score_ext, dev_step_diag_ext);
      cudaCheckKernel();
    }
    
    {
      dim3 k1_dimBlock(8);
      dim3 k1_dimGrid(1);
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_gap_size_v_ext);
      cudaCheckKernel();
    }
    
    {
      dim3 k2_dimBlock(8);
      dim3 k2_dimGrid(1);
      kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_best_gap_v_ext);
      cudaCheckKernel();
    }
    
    {
      dim3 k3_dimBlock(16);
      dim3 k3_dimGrid(1);
      kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_gap_size_h_ext);
      cudaCheckKernel();
    }
    
    {
      dim3 k4_dimBlock(16);
      dim3 k4_dimGrid(1);
      kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_best_gap_h_ext);
      cudaCheckKernel();
    }
    
    for (int c0 = 0; c0 <= 22; c0 += 1)
      {
        dim3 k5_dimBlock(16);
        dim3 k5_dimGrid(1);
        kernel5 <<<k5_dimGrid, k5_dimBlock>>> (dev_H, dev_H_ext, dev_best_gap_h_ext, dev_best_gap_v_ext, dev_bt, dev_gap_size_h_ext, dev_gap_size_v_ext, dev_sim_score_ext, dev_step_diag_ext, c0);
        cudaCheckKernel();
      }
      
    cudaCheckReturn(cudaMemcpy(H, dev_H, (17) * (9) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(bt, dev_bt, (16) * (8) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaFree(dev_H));
    cudaCheckReturn(cudaFree(dev_H_ext));
    cudaCheckReturn(cudaFree(dev_alt));
    cudaCheckReturn(cudaFree(dev_best_gap_h_ext));
    cudaCheckReturn(cudaFree(dev_best_gap_v_ext));
    cudaCheckReturn(cudaFree(dev_bt));
    cudaCheckReturn(cudaFree(dev_gap_size_h_ext));
    cudaCheckReturn(cudaFree(dev_gap_size_v_ext));
    cudaCheckReturn(cudaFree(dev_ref));
    cudaCheckReturn(cudaFree(dev_sim_score_ext));
    cudaCheckReturn(cudaFree(dev_step_diag_ext));
  }
}

//void dsa_kernel(char alt[M], char ref[N], int H[M + 1][N + 1], int bt[M][N]) {
//  char alt_ext[M][N];
//  char ref_ext[M][N];
//  int sim_score_ext[M][N];
//  int H_ext[M][N];
//
//  int step_diag_ext[M][N];
// 
//  int best_gap_v_ext[M][N];
//  int gap_size_v_ext[M][N];
//  int step_down_ext[M][N];
//  int kd_ext[M][N];
//  
//  int best_gap_h_ext[M][N];
//  int gap_size_h_ext[M][N];
//  int step_right_ext[M][N];
//  int ki_ext[M][N];
//
//  int sw_tmp1_ext[M][N];
//  int sw_tmp2_ext[M][N];
//  int bt_tmp1_ext[M][N];
//  int bt_tmp2_ext[M][N];
//
//  for (int i = 0; i < M; i++)
//    for (int j = 0; j < N; j++) {
//      // reuse at j-axis
//      if (j == 0) {
//        alt_ext[i][j] = alt[i];
//      } else {
//        alt_ext[i][j] = alt_ext[i][j - 1];
//      }
//
//      // reuse at i-axis
//      if (i == 0) {
//        ref_ext[i][j] = ref[j];
//      } else {
//        ref_ext[i][j] = ref_ext[i - 1][j];
//      }
//
//      sim_score_ext[i][j] = (alt_ext[i][j] == ref_ext[i][j])? P_W_MATCH: P_W_MISMATCH;
//
//      if (i == 0 || j == 0) {
//        step_diag_ext[i][j] = (MATRIX_MIN_CUTOFF > sim_score_ext[i][j])? MATRIX_MIN_CUTOFF: sim_score_ext[i][j];
//      } else {
//        step_diag_ext[i][j] = (MATRIX_MIN_CUTOFF > (H_ext[i - 1][j - 1] + sim_score_ext[i][j]))? MATRIX_MIN_CUTOFF: (H_ext[i - 1][j - 1] + sim_score_ext[i][j]);
//      }
//
//      if (i == 0) {
//        if (P_W_OPEN > P_W_EXTEND + LOW_INIT_VALUE) {
//          best_gap_v_ext[i][j] = P_W_OPEN;
//          gap_size_v_ext[i][j] = 1;
//        } else {
//          best_gap_v_ext[i][j] = P_W_EXTEND + LOW_INIT_VALUE;
//          gap_size_v_ext[i][j]++;
//        }
//      } else {
//        if (H_ext[i - 1][j] + P_W_OPEN > best_gap_v_ext[i - 1][j] + P_W_EXTEND) {
//          best_gap_v_ext[i][j] = H_ext[i - 1][j] + P_W_OPEN;
//          gap_size_v_ext[i][j] = 1;
//        } else {
//          best_gap_v_ext[i][j] = best_gap_v_ext[i - 1][j] + P_W_EXTEND;
//          gap_size_v_ext[i][j] = gap_size_v_ext[i - 1][j] + 1;
//        }
//      }
//
//      step_down_ext[i][j] = (MATRIX_MIN_CUTOFF > best_gap_v_ext[i][j])? MATRIX_MIN_CUTOFF: best_gap_v_ext[i][j];
//
//      kd_ext[i][j] = gap_size_v_ext[i][j];
//
//      if (j == 0) {
//        if (P_W_OPEN > P_W_EXTEND + LOW_INIT_VALUE) {
//          best_gap_h_ext[i][j] = P_W_OPEN;
//          gap_size_h_ext[i][j] = 1;
//        } else {
//          best_gap_h_ext[i][j] = P_W_EXTEND + LOW_INIT_VALUE;
//          gap_size_h_ext[i][j]++;
//        } 
//      } else {
//        if (H_ext[i][j - 1] + P_W_OPEN > best_gap_h_ext[i][j - 1] + P_W_EXTEND) {
//          best_gap_h_ext[i][j] = H_ext[i][j - 1] + P_W_OPEN;
//          gap_size_h_ext[i][j] = 1;
//        } else {
//          best_gap_h_ext[i][j] = best_gap_h_ext[i][j - 1] + P_W_EXTEND;
//          gap_size_h_ext[i][j] = gap_size_h_ext[i][j - 1] + 1;
//        }
//      }
//
//      step_right_ext[i][j] = (MATRIX_MIN_CUTOFF > best_gap_h_ext[i][j])? MATRIX_MIN_CUTOFF: best_gap_h_ext[i][j];
//        
//      ki_ext[i][j] = gap_size_h_ext[i][j];
//
//      sw_tmp1_ext[i][j] = max(step_diag_ext[i][j], step_down_ext[i][j]);
//      sw_tmp2_ext[i][j] = max(sw_tmp1_ext[i][j], step_right_ext[i][j]);
//      
//      H_ext[i][j] = sw_tmp2_ext[i][j];
//      H[i + 1][j + 1] = H_ext[i][j];
//
//      bt_tmp1_ext[i][j] = (H_ext[i][j] == step_right_ext[i][j])? -ki_ext[i][j] : kd_ext[i][j];
//      bt_tmp2_ext[i][j] = (H_ext[i][j] == step_diag_ext[i][j])? 0 : bt_tmp1_ext[i][j];
//
//      bt[i][j] = bt_tmp2_ext[i][j];
//    }
//}
