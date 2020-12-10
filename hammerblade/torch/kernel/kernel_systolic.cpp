//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#define BLOCK_DIM 12 // sqrt(4KB/4 byte/4 data matrix) = 15 max
#define SYSTOLIC_X_DIM 8
#define SYSTOLIC_Y_DIM 8
#include <kernel_common.hpp>
#include <kernel_addmm.hpp>
#include <kernel_circular_buffer.hpp>

namespace {

inline void spcpy(float* dest, float* src) {
  for (int i = 0; i < BLOCK_DIM * BLOCK_DIM; i += 8) {
        register float tmp0 = *(src + 0);
        register float tmp1 = *(src + 1);
        register float tmp2 = *(src + 2);
        register float tmp3 = *(src + 3);
        register float tmp4 = *(src + 4);
        register float tmp5 = *(src + 5);
        register float tmp6 = *(src + 6);
        register float tmp7 = *(src + 7);
        asm volatile("": : :"memory");
        *(dest + 0) = tmp0;
        *(dest + 1) = tmp1;
        *(dest + 2) = tmp2;
        *(dest + 3) = tmp3;
        *(dest + 4) = tmp4;
        *(dest + 5) = tmp5;
        *(dest + 6) = tmp6;
        *(dest + 7) = tmp7;
        src += 8;
        dest += 8;
  }
}

} // namespace

extern "C" {

  __attribute__ ((noinline))  int tensorlib_systolic(
          hb_tensor_t* _result,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2) {

    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);

    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);

    // calculate number of row and col blocks in each matrix
    int m1_num_blk_per_col = (r1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per col
    int m1_num_blk_per_row = (c1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per row
    int m2_num_blk_per_col = (r2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per col
    int m2_num_blk_per_row = (c2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per row

    // Config
    // 0 -- idle
    // 1 -- row DMA
    // 2 -- col DMA
    // 3 -- compute
    // 4 -- polyA - col
    // 5 -- polyA - row

    char systolic_6x14_gemm[8][16] = {
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
    };

    // Activate config
    char (&mc_config)[8][16] = systolic_6x14_gemm;
    char tile_config = mc_config[bsg_y][bsg_x];

    // buffers -- with double buffering
    float sp_result[BLOCK_DIM * BLOCK_DIM];
    float* sp_mat1;
    float* sp_mat2;
    float* sp_mat1_remote;
    float* sp_mat2_remote;

    CircularBuffer::FIFO<float, BLOCK_DIM * BLOCK_DIM, 2> mat1_fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);
    CircularBuffer::FIFO<float, BLOCK_DIM * BLOCK_DIM, 2> mat2_fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);

    bool should_pass_right = bsg_x == SYSTOLIC_X_DIM ? false : true;
    bool should_pass_down  = bsg_y == SYSTOLIC_Y_DIM ? false : true;

    auto compute_task = [&] () {

      for (int rr = bsg_y-1; rr < m1_num_blk_per_col; rr += SYSTOLIC_Y_DIM) {
        for (int rc = bsg_x-1; rc < m2_num_blk_per_row; rc += SYSTOLIC_X_DIM) {

          reset_sp(sp_result);

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            // wait until buffer is loaded
            sp_mat2 = mat2_fifo.obtain_rd_ptr();

            // wait until buffer is loaded
            sp_mat1 = mat1_fifo.obtain_rd_ptr();

            // do compute
            compute_simple(sp_result, sp_mat1, sp_mat2);

            // flag that we are done with the buffer
            mat1_fifo.finish_rd_ptr();
            mat2_fifo.finish_rd_ptr();
          }

          // write back
          sp_to_dram(result, sp_result, rr, rc);
        }
      }
    };

    auto col_dma_task = [&] () {
      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int rc = bsg_x-1; rc < m2_num_blk_per_row; rc += SYSTOLIC_X_DIM) {

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
          }

        }
      }
    };

    auto row_dma_task = [&] () {
      for (int rr = bsg_y-1; rr < m1_num_blk_per_col; rr += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            sp_mat2_remote = mat2_fifo.obtain_wr_ptr();
            dram_to_sp_simple_generic(sp_mat2_remote, mat2, mat1x, rc);
            mat2_fifo.finish_wr_ptr();
            sp_mat1_remote = mat1_fifo.obtain_wr_ptr();
            dram_to_sp_simple_generic(sp_mat1_remote, mat1, rr, mat1x);
            mat1_fifo.finish_wr_ptr();
          }

        }
      }
    };

    auto polyACol = [&]() {
      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            mat2_fifo.obtain_rd_ptr();
            mat2_fifo.finish_rd_ptr();
          }

        }
      }
    };

    auto polyARow = [&]() {
      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            mat1_fifo.obtain_rd_ptr();
            mat1_fifo.finish_rd_ptr();
          }

        }
      }
    };

    g_barrier.sync();

    bsg_cuda_print_stat_kernel_start();

    // schedule
    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        // row DMA
        row_dma_task();
        break;
      case 2:
        // col DMA
        col_dma_task();
        break;
      case 3:
        // PE
        compute_task();
        break;
      case 4:
        // PolyA Col
        polyACol();
        break;
      case 5:
        // PolyA Row
        polyARow();
        break;
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_systolic, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

