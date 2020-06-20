//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <kernel_addmm.hpp>
#define BLOCK_DIM 8 // sqrt(4KB/4 byte/4 data matrix) = 15 max

extern "C" {

static void dram_to_sp_simple(
          float* dest,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        int j = 0;
        for (;j < dim_x - 8; j += 8) {
            dest[row_offset + j]     = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            dest[row_offset + j + 1] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            dest[row_offset + j + 2] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            dest[row_offset + j + 3] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            dest[row_offset + j + 4] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            dest[row_offset + j + 5] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            dest[row_offset + j + 6] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            dest[row_offset + j + 7] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
        }
        // fixup
        for (;j < dim_x; j++) {
            dest[row_offset + j] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
        }
    }
}

static void dram_to_sp(
          float* dest,
          float coeff,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        int j = 0;
        for (;j < dim_x - 8; j += 8) {
            dest[row_offset + j] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            dest[row_offset + j + 1] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            dest[row_offset + j + 2] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            dest[row_offset + j + 3] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            dest[row_offset + j + 4] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            dest[row_offset + j + 5] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            dest[row_offset + j + 6] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            dest[row_offset + j + 7] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
        }
        // fixup
        for (;j < dim_x; j++) {
            dest[row_offset + j] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
        }
    }
}

  __attribute__ ((noinline))  int tensorlib_addmm(
          hb_tensor_t* _result,
          hb_tensor_t* _self,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2,
          float* _beta,
          float* _alpha) {

    auto self = HBTensor<float, 2>(_self);
    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);
    float beta = *_beta;
    float alpha = *_alpha;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();


    // v2: single tile, use blocking
    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);
    hb_assert(c1 == r2);

    // calculate number of row and col blocks in each matrix
    int m1_num_blk_per_row = (r1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per row
    int m1_num_blk_per_col = (c1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per col
    int m2_num_blk_per_row = (r2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per row
    int m2_num_blk_per_col = (c2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per col

    // calculate dimensions of the last row and col block in each matrix
    int m1_last_blk_dim_x = c1 % BLOCK_DIM == 0 ? BLOCK_DIM : c1 % BLOCK_DIM; // x dimension of last block of mat1
    int m1_last_blk_dim_y = r1 % BLOCK_DIM == 0 ? BLOCK_DIM : r1 % BLOCK_DIM; // y dimension of last block of mat1
    int m2_last_blk_dim_x = c2 % BLOCK_DIM == 0 ? BLOCK_DIM : c2 % BLOCK_DIM; // x dimension of last block of mat2
    int m2_last_blk_dim_y = r2 % BLOCK_DIM == 0 ? BLOCK_DIM : r2 % BLOCK_DIM; // y dimension of last block of mat2

    // iterate over result blocks
    hb_tiled_for(m1_num_blk_per_row * m2_num_blk_per_col, [&](size_t ridx) {
        int rr = ridx / m2_num_blk_per_col;
        // rc is index of col block in result matrix
        int rc = ridx % m2_num_blk_per_col;
        // calculate current result block dimensions
        int res_dim_y = rr == m1_num_blk_per_row - 1 ? m1_last_blk_dim_y : BLOCK_DIM;
        int res_dim_x = rc == m2_num_blk_per_col - 1 ? m2_last_blk_dim_x : BLOCK_DIM;
        int partial_block = (res_dim_y != BLOCK_DIM) || (res_dim_x != BLOCK_DIM);

        // initialize scratchpad result (load beta * self into result)

        // unrolled version
        float sp_result[res_dim_y * res_dim_x];
        float sp_self[res_dim_y * res_dim_x];
        dram_to_sp(sp_self, beta, self, res_dim_y, res_dim_x, rr, rc);
        memset(sp_result, 0, res_dim_y * res_dim_x * sizeof(float));
        // end: unrolled version

        // process mat1 and mat2 for this result block
        // only care about blocks of mat1 in row rr
        // and blocks of mat2 in col rc
        for (int mat1x = 0, mat2y = 0; mat1x < m1_num_blk_per_col && mat2y < m2_num_blk_per_row; mat1x++, mat2y++) {
            // calculate current block dimensions
            int mid_dim = mat1x == m1_num_blk_per_col - 1 ? m1_last_blk_dim_x : BLOCK_DIM;
            partial_block = partial_block || (mid_dim != BLOCK_DIM);

            // load mat1 and mat2 into scratchpad

            // unrolled version
            float sp_mat1[res_dim_y * mid_dim];
            float sp_mat2[mid_dim * res_dim_x];
            dram_to_sp_simple(sp_mat1, mat1, res_dim_y, mid_dim, rr, mat1x);
            dram_to_sp_simple(sp_mat2, mat2, mid_dim, res_dim_x, mat2y, rc);
            compute_simple(sp_result, sp_mat1, sp_mat2, res_dim_y, res_dim_x, mid_dim);
            // end: unrolled version

        }

        // copy this block back into DRAM
        for (int i = 0; i < res_dim_y; i++) {
            for (int j = 0; j < res_dim_x; j++) {
                // unrolled version
                result(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j) = sp_self[i * res_dim_x + j] + alpha * sp_result[i * res_dim_x + j];
                // end: unrolled version
            }
        }
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)

}

