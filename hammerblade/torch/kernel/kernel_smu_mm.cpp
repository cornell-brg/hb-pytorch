//====================================================================
// SMU-based systolic MM kernel
//====================================================================
// 12/01/2020 Lin Cheng, Peitian Pan
//            (lc873@cornell.edu, pp482@cornell.edu)

#define BLOCK_DIM 12 // sqrt(4KB/4 byte/4 data matrix) = 15 max
#define SYSTOLIC_X_DIM 16
#define SYSTOLIC_Y_DIM 8
#include <kernel_common.hpp>
#include <kernel_addmm.hpp>
#include <kernel_circular_buffer.hpp>
#include <hb_smu.hpp>

typedef CircularBuffer::FIFO<float, BLOCK_DIM*BLOCK_DIM, 2> DoubleBuffer;
typedef HBTensor<float, 2> MMTensor;

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

inline void
loop_inc(int rr, int rc, int k, int& rr_nxt, int& rc_nxt, int& k_nxt,
    int m1_num_blk_per_col, int m2_num_blk_per_row, int m1_num_blk_per_row) {
  rr_nxt = rr;
  rc_nxt = rc;
  k_nxt = k+1;
  if ( k_nxt == m1_num_blk_per_row ) {
    k_nxt = 0;
    rc_nxt = rc + SYSTOLIC_X_DIM;
    if ( rc_nxt >= m2_num_blk_per_row) {
      rc_nxt = bsg_x;
      rr_nxt = rr + SYSTOLIC_Y_DIM;
      if ( rr_nxt >= m1_num_blk_per_col ) {
        rr_nxt = bsg_y;
      }
    }
  }
}

// load block: only called by first col/row tiles
// NOTE: this function only works for double-buffering!

inline void
load_block(bool is_col, DoubleBuffer& buf, MMTensor& src, int* ack,
           int rr, int rc, int k, int m1_num_blk_per_col,
           int m2_num_blk_per_row, int m1_num_blk_per_row) {

  int rr_nxt, rc_nxt, k_nxt;

  loop_inc( rr, rc, k, rr_nxt, rc_nxt, k_nxt,
            m1_num_blk_per_col, m2_num_blk_per_row, m1_num_blk_per_row );

  int r_idx = is_col ? rr : k;
  int c_idx = is_col ? k : rc;

  int r_idx_nxt = is_col ? rr_nxt : k_nxt;
  int c_idx_nxt = is_col ? k_nxt : rc_nxt;

  bool is_last_block = k == (m1_num_blk_per_row);

  if ( !((rr == bsg_y) && (rc == bsg_x) && (k == 0)) ) {
    // If not first call, wait till SMU acks
    wait_smu( ack );
    if ( ! is_last_block )
      // One FIFO has not been loaded; call SMU to load data
      launch_smu_mm( is_col, src, r_idx_nxt, c_idx_nxt, buf.get_next_buffer(), ack );
  } else {
    // No FIFO was set to load data -- only possible at the first call
    // In this case we load two blocks and wait til the first block
    // comes back.
    launch_smu_mm( is_col, src, r_idx, c_idx, buf.get_buffer(), ack );
    wait_smu( ack );
    // Load second block. Wait for ACK in the next call
    launch_smu_mm( is_col, src, r_idx_nxt, c_idx_nxt, buf.get_next_buffer(), ack );
  }

  // Tell the buffer that the SMU has finished a pull-based write
  buf.SMU_finish_wb();

}

} // namespace

extern "C" {

  __attribute__ ((noinline))  int tensorlib_smu_mm(
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

    // buffers -- with double buffering
    float sp_result[BLOCK_DIM * BLOCK_DIM];
    float* sp_mat1;
    float* sp_mat2;
    float* sp_mat1_remote;
    float* sp_mat2_remote;

    DoubleBuffer mat1_fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);
    DoubleBuffer mat2_fifo(bsg_y-1, bsg_x, bsg_y+1, bsg_x);

    bool is_first_row      = bsg_y == 0                ? true  : false;
    bool is_first_col      = bsg_x == 0                ? true  : false;
    bool should_pass_right = bsg_x == SYSTOLIC_X_DIM-1 ? false : true;
    bool should_pass_down  = bsg_y == SYSTOLIC_Y_DIM-1 ? false : true;

    // ACKs from SMU(s)
    int ack_row = 0;
    int ack_col = 0;

    // Start kernel

    g_barrier.sync();

    bsg_cuda_print_stat_kernel_start();

    // triple nested loops but at the block level
    for (int rr = bsg_y; rr < m1_num_blk_per_col; rr += SYSTOLIC_Y_DIM) {
      for (int rc = bsg_x; rc < m2_num_blk_per_row; rc += SYSTOLIC_X_DIM) {

        reset_sp(sp_result); // this sets BLOCK_DIM*BLOCK_DIM matrix to be 0

        for (int mat1x = 0; mat1x < m1_num_blk_per_row; mat1x++) {
          // Begin per-block computation

          if ( is_first_row )
            // load block at first row
            load_block( false, mat2_fifo, mat2, &ack_row, rr, rc, mat1x,
                        m1_num_blk_per_col, m2_num_blk_per_row,
                        m1_num_blk_per_row );

          if ( is_first_col )
            // load block at first column
            load_block( true, mat1_fifo, mat1, &ack_col, rr, rc, mat1x,
                        m1_num_blk_per_col, m2_num_blk_per_row,
                        m1_num_blk_per_row );

          // wait until buffer is loaded
          sp_mat2 = mat2_fifo.obtain_rd_ptr();

          if ( should_pass_down && rr+1 < m1_num_blk_per_col ) {
            sp_mat2_remote = mat2_fifo.obtain_wr_ptr();
            // copy mat2 to S
            spcpy(sp_mat2_remote, sp_mat2);
            mat2_fifo.finish_wr_ptr();
          }

          // wait until buffer is loaded
          sp_mat1 = mat1_fifo.obtain_rd_ptr();

          if (should_pass_right && rc+1 < m2_num_blk_per_row) {
            sp_mat1_remote = mat1_fifo.obtain_wr_ptr();
            // copy mat1 to E
            spcpy(sp_mat1_remote, sp_mat1);
            mat1_fifo.finish_wr_ptr();
          }

          // do compute. use BLOCK_DIM
          compute_simple(sp_result, sp_mat1, sp_mat2);

          // flag that we are done with the buffer
          if ( is_first_col )
            mat1_fifo.SMU_finish_rd();
          else
            mat1_fifo.finish_rd_ptr();
          if ( is_first_row )
            mat2_fifo.SMU_finish_rd();
          else
            mat2_fifo.finish_rd_ptr();
        }

        // write back. use BLOCK_DIM
        sp_to_dram(result, sp_result, rr, rc);
      }
    }

    // End kernel

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_smu_mm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
