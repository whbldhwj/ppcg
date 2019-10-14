#include "kernel_kernel.hu"
__global__ void kernel0(char *alt, char *ref, int *sim_score_ext, int *step_diag_ext)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ char shared_alt[16];
    __shared__ char shared_ref[8];
    int private_sim_score_ext[1][1];

    {
      if (t0 == 0) {
        for (int c0 = t1; c0 <= 15; c0 += 8)
          shared_alt[c0] = alt[c0];
        shared_ref[t1] = ref[t1];
      }
      __syncthreads();
      private_sim_score_ext[0][0] = ((shared_alt[t0] == shared_ref[t1]) ? 200 : (-150));
      if (t0 == 0) {
        step_diag_ext[0 * 8 + t1] = (((-100000000) > private_sim_score_ext[0][0]) ? (-100000000) : private_sim_score_ext[0][0]);
      } else if (t1 == 0) {
        step_diag_ext[t0 * 8 + 0] = (((-100000000) > private_sim_score_ext[0][0]) ? (-100000000) : private_sim_score_ext[0][0]);
      }
      if (t0 >= 1 && t1 >= 1)
        sim_score_ext[t0 * 8 + t1] = private_sim_score_ext[0][0];
    }
}
__global__ void kernel1(int *gap_size_v_ext)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_gap_size_v_ext[1][1];

    {
      private_gap_size_v_ext[0][0] = 1;
      gap_size_v_ext[0 * 8 + t0] = private_gap_size_v_ext[0][0];
    }
}
__global__ void kernel2(int *best_gap_v_ext)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_best_gap_v_ext[1][1];

    {
      private_best_gap_v_ext[0][0] = (-260);
      best_gap_v_ext[0 * 8 + t0] = private_best_gap_v_ext[0][0];
    }
}
__global__ void kernel3(int *gap_size_h_ext)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_gap_size_h_ext[1][1];

    {
      private_gap_size_h_ext[0][0] = 1;
      gap_size_h_ext[t0 * 8 + 0] = private_gap_size_h_ext[0][0];
    }
}
__global__ void kernel4(int *best_gap_h_ext)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_best_gap_h_ext[1][1];

    {
      private_best_gap_h_ext[0][0] = (-260);
      best_gap_h_ext[t0 * 8 + 0] = private_best_gap_h_ext[0][0];
    }
}
__global__ void kernel5(int *H, int *H_ext, int *best_gap_h_ext, int *best_gap_v_ext, int *bt, int *gap_size_h_ext, int *gap_size_v_ext, int *sim_score_ext, int *step_diag_ext, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_H_ext_6[1][1];
    int private_bt_tmp1_ext[1][1];
    int private_bt_tmp2_ext[1][1];
    int private_kd_ext[1][1];
    int private_ki_ext[1][1];
    int private_step_diag_ext[1][1];
    int private_step_down_ext[1][1];
    int private_step_right_ext[1][1];
    int private_sw_tmp1_ext[1][1];
    int private_sw_tmp2_ext[1][1];

    if (c0 >= t0 && t0 + 7 >= c0) {
      if ((t0 >= 1 && c0 == t0) || t0 == 0) {
        private_step_diag_ext[0][0] = step_diag_ext[t0 * 8 + (-t0 + c0)];
      } else {
        private_step_diag_ext[0][0] = (((-100000000) > (H_ext[(t0 - 1) * 8 + (-t0 + c0 - 1)] + sim_score_ext[t0 * 8 + (-t0 + c0)])) ? (-100000000) : (H_ext[(t0 - 1) * 8 + (-t0 + c0 - 1)] + sim_score_ext[t0 * 8 + (-t0 + c0)]));
      }
      if (c0 >= t0 + 1)
        if ((H_ext[t0 * 8 + (-t0 + c0 - 1)] + (-260)) > (best_gap_h_ext[t0 * 8 + (-t0 + c0 - 1)] + (-11))) {
          best_gap_h_ext[t0 * 8 + (-t0 + c0)] = (H_ext[t0 * 8 + (-t0 + c0 - 1)] + (-260));
          gap_size_h_ext[t0 * 8 + (-t0 + c0)] = 1;
        } else {
          best_gap_h_ext[t0 * 8 + (-t0 + c0)] = (best_gap_h_ext[t0 * 8 + (-t0 + c0 - 1)] + (-11));
          gap_size_h_ext[t0 * 8 + (-t0 + c0)] = (gap_size_h_ext[t0 * 8 + (-t0 + c0 - 1)] + 1);
        }
      if (t0 >= 1)
        if ((H_ext[(t0 - 1) * 8 + (-t0 + c0)] + (-260)) > (best_gap_v_ext[(t0 - 1) * 8 + (-t0 + c0)] + (-11))) {
          best_gap_v_ext[t0 * 8 + (-t0 + c0)] = (H_ext[(t0 - 1) * 8 + (-t0 + c0)] + (-260));
          gap_size_v_ext[t0 * 8 + (-t0 + c0)] = 1;
        } else {
          best_gap_v_ext[t0 * 8 + (-t0 + c0)] = (best_gap_v_ext[(t0 - 1) * 8 + (-t0 + c0)] + (-11));
          gap_size_v_ext[t0 * 8 + (-t0 + c0)] = (gap_size_v_ext[(t0 - 1) * 8 + (-t0 + c0)] + 1);
        }
      private_step_down_ext[0][0] = (((-100000000) > best_gap_v_ext[t0 * 8 + (-t0 + c0)]) ? (-100000000) : best_gap_v_ext[t0 * 8 + (-t0 + c0)]);
      private_kd_ext[0][0] = gap_size_v_ext[t0 * 8 + (-t0 + c0)];
      private_step_right_ext[0][0] = (((-100000000) > best_gap_h_ext[t0 * 8 + (-t0 + c0)]) ? (-100000000) : best_gap_h_ext[t0 * 8 + (-t0 + c0)]);
      private_ki_ext[0][0] = gap_size_h_ext[t0 * 8 + (-t0 + c0)];
      private_sw_tmp1_ext[0][0] = ((private_step_diag_ext[0][0] > private_step_down_ext[0][0]) ? private_step_diag_ext[0][0] : private_step_down_ext[0][0]);
      private_sw_tmp2_ext[0][0] = ((private_sw_tmp1_ext[0][0] > private_step_right_ext[0][0]) ? private_sw_tmp1_ext[0][0] : private_step_right_ext[0][0]);
      private_H_ext_6[0][0] = private_sw_tmp2_ext[0][0];
      H[(t0 + 1) * 9 + (-t0 + c0 + 1)] = private_H_ext_6[0][0];
      private_bt_tmp1_ext[0][0] = ((private_H_ext_6[0][0] == private_step_right_ext[0][0]) ? (-private_ki_ext[0][0]) : private_kd_ext[0][0]);
      private_bt_tmp2_ext[0][0] = ((private_H_ext_6[0][0] == private_step_diag_ext[0][0]) ? 0 : private_bt_tmp1_ext[0][0]);
      bt[t0 * 8 + (-t0 + c0)] = private_bt_tmp2_ext[0][0];
      if (c0 <= 21)
        H_ext[t0 * 8 + (-t0 + c0)] = private_H_ext_6[0][0];
    }
}
