//====================================================================
// MM kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <kernel_cgra_common.hpp>

#include "cgra_kernel_cfgs/cgra_cfg_test_os_gemm_fp32.dat"

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mm(
          hb_tensor_t* _result,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2) {

    /* return 0; */

    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    if ( (__bsg_x == 0) && (__bsg_y == 0) ) {
      // use CGRA xcel
      int r1 = mat1.dim(0);
      int c1 = mat1.dim(1);
      int r2 = mat2.dim(0);
      int c2 = mat2.dim(1);
      hb_assert(c1 == r2);

      // PP: print out the dimension of 2D tensors
      /* bsg_print_hexadecimal(r1); */
      /* bsg_print_hexadecimal(c1); */
      /* bsg_print_hexadecimal(r2); */
      /* bsg_print_hexadecimal(c2); */

      int m1_num_blk_dim_y = (r1 + CGRA_MM_BLOCK_SIZE - 1) / CGRA_MM_BLOCK_SIZE;
      int m1_num_blk_dim_x = (c1 + CGRA_MM_BLOCK_SIZE - 1) / CGRA_MM_BLOCK_SIZE;
      int m2_num_blk_dim_y = (r2 + CGRA_MM_BLOCK_SIZE - 1) / CGRA_MM_BLOCK_SIZE;
      int m2_num_blk_dim_x = (c2 + CGRA_MM_BLOCK_SIZE - 1) / CGRA_MM_BLOCK_SIZE;

      int m1_last_blk_dim_x = c1 % CGRA_MM_BLOCK_SIZE == 0 ? CGRA_MM_BLOCK_SIZE : c1 % CGRA_MM_BLOCK_SIZE;
      int m1_last_blk_dim_y = r1 % CGRA_MM_BLOCK_SIZE == 0 ? CGRA_MM_BLOCK_SIZE : r1 % CGRA_MM_BLOCK_SIZE;
      int m2_last_blk_dim_x = c2 % CGRA_MM_BLOCK_SIZE == 0 ? CGRA_MM_BLOCK_SIZE : c2 % CGRA_MM_BLOCK_SIZE;
      int m2_last_blk_dim_y = r2 % CGRA_MM_BLOCK_SIZE == 0 ? CGRA_MM_BLOCK_SIZE : r2 % CGRA_MM_BLOCK_SIZE;

      // Only do 1/16 of the overall workload.
      /* int rr_inc = 16; */

      // Do the entire workload
      int rr_inc = 1;

      // Stage configuration
      stage_cfg(cgra_mm_cfg, cgra_mm_cfg_size);

      // iterate over all blocks in the result matrix C
      for (int rr = 0; rr < m1_num_blk_dim_y; rr += rr_inc) {
        for (int rc = 0; rc < m2_num_blk_dim_x; rc++) {
          int is_mat1_dim_y_partial = (rr == m1_num_blk_dim_y - 1) && (m1_last_blk_dim_y != CGRA_MM_BLOCK_SIZE);
          int is_mat2_dim_x_partial = (rc == m2_num_blk_dim_x - 1) && (m2_last_blk_dim_x != CGRA_MM_BLOCK_SIZE);

          // Initialize this 32x32 result block to be 0.0
          bsg_remote_float_ptr xcel_base_ptr = reinterpret_cast<bsg_remote_float_ptr>(bsg_global_ptr(XCEL_X_CORD, XCEL_Y_CORD, 0));
          bsg_remote_float_ptr xcel_mat1_ptr = xcel_base_ptr + cgra_mm_cfg_size;
          bsg_remote_float_ptr xcel_mat2_ptr = xcel_mat1_ptr + CGRA_MM_BLOCK_SIZE*CGRA_MM_BLOCK_SIZE;
          bsg_remote_float_ptr xcel_result_ptr = xcel_mat2_ptr + CGRA_MM_BLOCK_SIZE*CGRA_MM_BLOCK_SIZE;

          for (int i_c = 0; i_c < CGRA_MM_BLOCK_SIZE*CGRA_MM_BLOCK_SIZE; i_c++)
            *(xcel_result_ptr + i_c) = 0.0;

          // Iterate over the column block of A and row block of B
          for (int rk = 0; rk < m1_num_blk_dim_x; rk++) {
            // Offload this 32x32 @ 32x32 MM to CGRA
            // Block A: rr, rk
            // Block B: rk, rc
            int is_mat1_dim_x_partial = (rk == m1_num_blk_dim_x - 1) && (m1_last_blk_dim_x != CGRA_MM_BLOCK_SIZE);
            int is_mat2_dim_y_partial = (rk == m2_num_blk_dim_y - 1) && (m2_last_blk_dim_y != CGRA_MM_BLOCK_SIZE);

            // First stage mat1 and mat2 into scratchpad
            for (int ii = 0; ii < CGRA_MM_BLOCK_SIZE; ii++)
              for (int jj = 0; jj < CGRA_MM_BLOCK_SIZE; jj++) {
                int m1i = rr * CGRA_MM_BLOCK_SIZE + ii;
                int m1j = rk * CGRA_MM_BLOCK_SIZE + jj;
                int m2i = rk * CGRA_MM_BLOCK_SIZE + ii;
                int m2j = rc * CGRA_MM_BLOCK_SIZE + jj;

                int mat1_pad_y = is_mat1_dim_y_partial && (ii >= m1_last_blk_dim_y);
                int mat1_pad_x = is_mat1_dim_x_partial && (jj >= m1_last_blk_dim_x);
                int mat2_pad_y = is_mat2_dim_y_partial && (ii >= m2_last_blk_dim_y);
                int mat2_pad_x = is_mat2_dim_x_partial && (jj >= m2_last_blk_dim_x);

                *(xcel_mat1_ptr + ii*CGRA_MM_BLOCK_SIZE + jj) = (mat1_pad_y || mat1_pad_x) ? 0.0 : mat1(m1i, m1j);
                *(xcel_mat2_ptr + ii*CGRA_MM_BLOCK_SIZE + jj) = (mat2_pad_y || mat2_pad_x) ? 0.0 : mat2(m2i, m2j);
              }

            // Launch CGRA
            int cfg_IC = 0, done = 0;

            for (int i = 0; i < cgra_mm_instruction_size; i++)
              if (cgra_mm_instructions[i] == CONFIG_INST) {
                execute_config(
                    cgra_mm_arg0[cfg_IC], cgra_mm_arg1[cfg_IC],
                    cgra_mm_arg2[cfg_IC], cgra_mm_arg3[cfg_IC]
                );
                cfg_IC++;
              } else if (cgra_mm_instructions[i] == LAUNCH_INST) {
                execute_launch();
              }

            while (done == 0)
              CGRA_REG_RD(CGRA_REG_CALC_DONE, done);
          }

          // Write this result block back to DRAM
          int update_dim_y_size = is_mat1_dim_y_partial ? m1_last_blk_dim_y : CGRA_MM_BLOCK_SIZE;
          int update_dim_x_size = is_mat2_dim_x_partial ? m2_last_blk_dim_x : CGRA_MM_BLOCK_SIZE;
          for (int ii = 0; ii < update_dim_y_size; ii++)
            for (int jj = 0; jj < update_dim_x_size; jj++)
              result(rr * CGRA_MM_BLOCK_SIZE + ii, rc * CGRA_MM_BLOCK_SIZE + jj) = *(xcel_result_ptr + ii*CGRA_MM_BLOCK_SIZE + jj);
        }
      }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

