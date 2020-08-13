//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#define BLOCK_DIM 8 // sqrt(4KB/4 byte/4 data matrix) = 15 max
#include <kernel_common.hpp>
#include <kernel_addmm.hpp>

template <typename FuncInit, typename FuncMain, typename FuncWB>
inline void gemm_main_loop(HBTensor<float, 2> mat1,
                           HBTensor<float, 2> mat2,
                           FuncInit tile_init,
                           FuncMain tile_task,
                           FuncWB   tile_finish) {

    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);

    // calculate number of row and col blocks in each matrix
    int m1_num_blk_per_col = (r1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per row
    int m1_num_blk_per_row = (c1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per col
    int m2_num_blk_per_col = (r2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per row
    int m2_num_blk_per_row = (c2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per col

    // calculate dimensions of the last row and col block in each matrix
    int m1_last_blk_dim_x = c1 % BLOCK_DIM == 0 ? BLOCK_DIM : c1 % BLOCK_DIM; // x dimension of last block of mat1
    int m1_last_blk_dim_y = r1 % BLOCK_DIM == 0 ? BLOCK_DIM : r1 % BLOCK_DIM; // y dimension of last block of mat1
    int m2_last_blk_dim_x = c2 % BLOCK_DIM == 0 ? BLOCK_DIM : c2 % BLOCK_DIM; // x dimension of last block of mat2
    int m2_last_blk_dim_y = r2 % BLOCK_DIM == 0 ? BLOCK_DIM : r2 % BLOCK_DIM; // y dimension of last block of mat2

    for (int i = 0; i < m1_num_blk_per_col; i += BSG_TILE_GROUP_Y_DIM) {
      for (int j = 0; j < m2_num_blk_per_row; j += BSG_TILE_GROUP_X_DIM) {
        int rr = i + __bsg_y;
        int rc = j + __bsg_x;
        int res_dim_y = rr == m1_num_blk_per_col - 1 ? m1_last_blk_dim_y : BLOCK_DIM;
        int res_dim_x = rc == m2_num_blk_per_row - 1 ? m2_last_blk_dim_x : BLOCK_DIM;
        int partial_block = (res_dim_y != BLOCK_DIM) || (res_dim_x != BLOCK_DIM);

        // init code for each output block
        tile_init();

        for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            int mid_dim = mat1x == m1_num_blk_per_row - 1 ? m1_last_blk_dim_x : BLOCK_DIM;
            partial_block = partial_block || (mid_dim != BLOCK_DIM);

            // main task
            tile_task(rr, rc, mat1x, res_dim_x, res_dim_y, mid_dim, partial_block);
        }

        // finishing up code for each output block
        tile_finish(rr, rc, res_dim_x, res_dim_y);
      }
    }
}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_systolic(
          hb_tensor_t* _result,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2) {

    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);

    // buffers -- with double buffering
    float sp_mat1_A[BLOCK_DIM * BLOCK_DIM];
    float sp_mat2_A[BLOCK_DIM * BLOCK_DIM];
    float sp_mat1_B[BLOCK_DIM * BLOCK_DIM];
    float sp_mat2_B[BLOCK_DIM * BLOCK_DIM];
    float *sp_mat1 = sp_mat1_A;
    float *sp_mat2 = sp_mat2_A;
    float sp_result[BLOCK_DIM * BLOCK_DIM];

    bsg_cuda_print_stat_kernel_start();

    auto tile_init = [&] { reset_sp(sp_result); };
    auto tile_task = [&] (int rr, int rc, int mat1x, int res_dim_x, int res_dim_y, int mid_dim, int partial_block) {
                          if (partial_block) { // general case
                            dram_to_sp(sp_mat1, mat1, res_dim_y, mid_dim, rr, mat1x);
                            dram_to_sp(sp_mat2, mat2, mid_dim, res_dim_x, mat1x, rc);
                            compute(sp_result, sp_mat1, sp_mat2, res_dim_y, res_dim_x, mid_dim);
                          } else {
                            dram_to_sp_simple(sp_mat1, mat1, rr, mat1x);
                            dram_to_sp_simple(sp_mat2, mat2, mat1x, rc);
                            compute_simple(sp_result, sp_mat1, sp_mat2);
                          }
                          // switch buffer
                          if (sp_mat1 == sp_mat1_A) {
                            sp_mat1 = sp_mat1_B;
                            sp_mat2 = sp_mat2_B;
                          } else {
                            sp_mat1 = sp_mat1_A;
                            sp_mat2 = sp_mat2_A;
                          }
                      };
    auto tile_finish = [&] (int rr, int rc, int res_dim_x, int res_dim_y) {
                            // copy this block back into DRAM
                            for (int i = 0; i < res_dim_y; i++) {
                              for (int j = 0; j < res_dim_x; j++) {
                                result(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j) = sp_result[i * res_dim_x + j];
                              }
                            }
                        };
    // schedule
    gemm_main_loop(mat1, mat2, tile_init, tile_task, tile_finish);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_systolic, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

