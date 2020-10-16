//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#define BLOCK_DIM 12 // sqrt(4KB/4 byte/4 data matrix) = 15 max
#define SYSTOLIC_X_DIM 14
#define SYSTOLIC_Y_DIM 6
#include <kernel_common.hpp>
#include <kernel_addmm.hpp>
#include <kernel_circular_buffer.hpp>

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
      {0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
      {0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0},
    //  {0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {1, 3, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {1, 3, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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

    CircularBuffer::FIFO<float, BLOCK_DIM * BLOCK_DIM, 1> mat1_fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);
    CircularBuffer::FIFO<float, BLOCK_DIM * BLOCK_DIM, 1> mat2_fifo(bsg_y-1, bsg_x, bsg_y+1, bsg_x);

    auto compute_task = [&] () {

      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          int rr = i + __bsg_y - 1;
          int rc = j + __bsg_x - 1;

          reset_sp(sp_result);

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            // wait until buffer is loaded
            sp_mat2 = mat2_fifo.obtain_rd_ptr();

            sp_mat2_remote = mat2_fifo.obtain_wr_ptr();
            // copy mat2 to S
            spcpy(sp_mat2_remote, sp_mat2);
            mat2_fifo.finish_wr_ptr();

            // wait until buffer is loaded
            sp_mat1 = mat1_fifo.obtain_rd_ptr();

            sp_mat1_remote = mat1_fifo.obtain_wr_ptr();
            // copy mat2 to S
            spcpy(sp_mat1_remote, sp_mat1);
            mat1_fifo.finish_wr_ptr();


            // do compute
            compute_simple(sp_result, sp_mat1, sp_mat2);

            // flag that we are done with the buffer
            mat1_fifo.finish_rd_ptr();
            mat2_fifo.finish_rd_ptr();
          }

          // write back
          for (int i = 0; i < BLOCK_DIM; i++) {
            for (int j = 0; j < BLOCK_DIM; j++) {
              result(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j) = sp_result[i * BLOCK_DIM + j];
            }
          }

        }
      }
    };

    auto col_dma_task = [&] () {
      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          int rc = j + __bsg_x - 1;

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            sp_mat2_remote = mat2_fifo.obtain_wr_ptr();
            dram_to_sp_simple(sp_mat2_remote, mat2, mat1x, rc);
            mat2_fifo.finish_wr_ptr();
          }

        }
      }
    };

    auto row_dma_task = [&] () {
      for (int i = 0; i < m1_num_blk_per_col; i += SYSTOLIC_Y_DIM) {
        for (int j = 0; j < m2_num_blk_per_row; j += SYSTOLIC_X_DIM) {

          int rr = i + __bsg_y - 1;

          for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
            sp_mat1_remote = mat1_fifo.obtain_wr_ptr();
            dram_to_sp_simple(sp_mat1_remote, mat1, rr, mat1x);
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


    // if (__bsg_id == 0 || __bsg_x > SYSTOLIC_X_DIM || __bsg_y > SYSTOLIC_Y_DIM) {
    //   // do nothing
    // } else if (__bsg_x == 0 && __bsg_y != 0) {
    //   // row DMA
    //   gemm_main_loop(mat1, mat2, __bsg_x, __bsg_y-1, [] {}, row_dma_task, [] (int rr, int rc) {});
    // } else if (__bsg_y == 0 && __bsg_x != 0) {
    //   // col DMA
    //   gemm_main_loop(mat1, mat2, __bsg_x-1, __bsg_y, [] {}, col_dma_task, [] (int rr, int rc) {});
    // } else {
    //   // PE
    //   gemm_main_loop(mat1, mat2, __bsg_x-1, __bsg_y-1, tile_init, tile_task, tile_finish);
    // }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_systolic, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

