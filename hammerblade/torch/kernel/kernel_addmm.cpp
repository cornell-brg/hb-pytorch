//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_DIM 8 // sqrt(4KB/4 byte/4 data matrix) = 15 max

extern "C" {

    // TODO:
    // have two versions of compute and dram_to_sp
    // one without if checks and one with

  void compute_simple(
          float* dest,
          float* sp_mat1,
          float* sp_mat2,
          float alpha,
          int dim_y,
          int dim_x,
          int mid_dim) {
    for (int i = 0; i < dim_y; i++) {
        int dest_row_offset = i * dim_x;
        int mat1_row_offset = i * mid_dim;
        for (int j = 0; j < dim_x; j++) {
            for (int k = 0; k < mid_dim; k += 8) {
                int mat1_idx = mat1_row_offset + k;
                int mat2_idx = k * dim_x + j;
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx] * sp_mat2[mat2_idx]; 
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 1] * sp_mat2[mat2_idx + dim_x];
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 2] * sp_mat2[mat2_idx + 2 * dim_x]; 
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 3] * sp_mat2[mat2_idx + 3 * dim_x];
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 4] * sp_mat2[mat2_idx + 4 * dim_x]; 
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 5] * sp_mat2[mat2_idx + 5 * dim_x]; 
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 6] * sp_mat2[mat2_idx + 6 * dim_x]; 
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 7] * sp_mat2[mat2_idx + 7 * dim_x]; 
            }
        }
    }
}

  void dram_to_sp_simple(
          float* dest,
          float coeff,
          HBTensor<float> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        for (int j = 0; j < dim_x; j += 8) {
            dest[row_offset + j] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            dest[row_offset + j + 1] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            dest[row_offset + j + 2] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            dest[row_offset + j + 3] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            dest[row_offset + j + 4] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            dest[row_offset + j + 5] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            dest[row_offset + j + 6] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            dest[row_offset + j + 7] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
        }
    }
}

  void compute(
          float* dest,
          float* sp_mat1,
          float* sp_mat2,
          float alpha,
          int dim_y,
          int dim_x,
          int mid_dim) {
    for (int i = 0; i < dim_y; i++) {
        int dest_row_offset = i * dim_x;
        int mat1_row_offset = i * mid_dim;
        for (int j = 0; j < dim_x; j++) {
            for (int k = 0; k < mid_dim; k += 8) {
                int mat1_idx = mat1_row_offset + k;
                int mat2_idx = k * dim_x + j;
                dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx] * sp_mat2[mat2_idx]; 
                if (k + 1 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 1] * sp_mat2[mat2_idx + dim_x];
                if (k + 2 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 2] * sp_mat2[mat2_idx + 2 * dim_x]; 
                if (k + 3 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 3] * sp_mat2[mat2_idx + 3 * dim_x];
                if (k + 4 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 4] * sp_mat2[mat2_idx + 4 * dim_x]; 
                if (k + 5 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 5] * sp_mat2[mat2_idx + 5 * dim_x]; 
                if (k + 6 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 6] * sp_mat2[mat2_idx + 6 * dim_x]; 
                if (k + 7 < mid_dim) dest[dest_row_offset + j] += alpha * sp_mat1[mat1_idx + 7] * sp_mat2[mat2_idx + 7 * dim_x]; 
            }
        }
    }
}

  void dram_to_sp(
          float* dest,
          float coeff,
          HBTensor<float> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        for (int j = 0; j < dim_x; j += 8) {
            dest[row_offset + j] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            if (j + 1 < dim_x) dest[row_offset + j + 1] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            if (j + 2 < dim_x) dest[row_offset + j + 2] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            if (j + 3 < dim_x) dest[row_offset + j + 3] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            if (j + 4 < dim_x) dest[row_offset + j + 4] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            if (j + 5 < dim_x) dest[row_offset + j + 5] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            if (j + 6 < dim_x) dest[row_offset + j + 6] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            if (j + 7 < dim_x) dest[row_offset + j + 7] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
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

    if (__bsg_id == 0) {

        auto self = HBTensor<float>(_self);
        auto mat1 = HBTensor<float>(_mat1);
        auto mat2 = HBTensor<float>(_mat2);
        auto result = HBTensor<float>(_result);
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
        for (int rr = 0; rr < m1_num_blk_per_row; rr++) { // rr is index of row block in result matrix
            for (int rc = 0; rc < m2_num_blk_per_col; rc++) { // rc is index of col block in result matrix
                // calculate current result block dimensions
                int res_dim_y = rr == m1_num_blk_per_row - 1 ? m1_last_blk_dim_y : BLOCK_DIM;
                int res_dim_x = rc == m2_num_blk_per_col - 1 ? m2_last_blk_dim_x : BLOCK_DIM;
                int partial_block = (res_dim_y != BLOCK_DIM) || (res_dim_x != BLOCK_DIM);

                // initialize scratchpad result (load beta * self into result)

                // unrolled version
                float sp_result[res_dim_y * res_dim_x];
                if (partial_block) {
                    dram_to_sp(sp_result, beta, self, res_dim_y, res_dim_x, rr, rc);
                } else {
                    dram_to_sp_simple(sp_result, beta, self, res_dim_y, res_dim_x, rr, rc);
                }
                // end: unrolled version

/*
                // original non-unrolled version
                float sp_result[res_dim_y][res_dim_x];

                for (int i = 0; i < res_dim_y; i++) {
                    for (int j = 0; j < res_dim_x; j++) {
                        sp_result[i][j] = beta * self(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j);
                    }
                }
                // end: original non-unrolled version
*/


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
                    if (partial_block) {
                        dram_to_sp(sp_mat1, 1.0f, mat1, res_dim_y, mid_dim, rr, mat1x);
                        dram_to_sp(sp_mat2, 1.0f, mat2, mid_dim, res_dim_x, mat2y, rc);
                        compute(sp_result, sp_mat1, sp_mat2, alpha, res_dim_y, res_dim_x, mid_dim);
                    } else {
                        dram_to_sp_simple(sp_mat1, 1.0f, mat1, res_dim_y, mid_dim, rr, mat1x);
                        dram_to_sp_simple(sp_mat2, 1.0f, mat2, mid_dim, res_dim_x, mat2y, rc);
                        compute_simple(sp_result, sp_mat1, sp_mat2, alpha, res_dim_y, res_dim_x, mid_dim);
                    }
                    // end: unrolled version


/*
                    // original non-unrolled version
                    float sp_mat1[res_dim_y][mid_dim];
                    float sp_mat2[mid_dim][res_dim_x];
                    for (int i = 0; i < res_dim_y; i++) {
                        for (int j = 0; j < mid_dim; j++) {
                            sp_mat1[i][j] = mat1(rr * BLOCK_DIM + i, mat1x * BLOCK_DIM + j);
                        }
                    }

                    for (int i = 0; i < mid_dim; i++) {
                        for (int j = 0; j < res_dim_x; j++) {
                            sp_mat2[i][j] = mat2(mat2y * BLOCK_DIM + i, rc * BLOCK_DIM + j);
                        }
                    }

                    // compute mat1 X mat2 for this block
                    for (int i = 0; i < res_dim_y; i++) {
                        for (int j = 0; j < res_dim_x; j++) {
                            for (int k = 0; k < mid_dim; k++) {
                                sp_result[i][j] += alpha * sp_mat1[i][k] * sp_mat2[k][j]; 
                            }
                        }
                    }
                    // end: non-unrolled version
*/

                }


                // copy this block back into DRAM
                for (int i = 0; i < res_dim_y; i++) {
                    for (int j = 0; j < res_dim_x; j++) {
                        // unrolled version
                        result(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j) = sp_result[i * res_dim_x + j];
                        // end: unrolled version

                        // original non-unrolled version
                        //result(rr * BLOCK_DIM + i, rc * BLOCK_DIM + j) = sp_result[i][j];
                        // end: original non-unrolled version
                    }
                }
            }
        }


/*
        // v1: naive version, single tile, access DRAM elm by elm
  
        int r1 = mat1.dim(0);
        int c1 = mat1.dim(1);
        int r2 = mat2.dim(0);
        int c2 = mat2.dim(1);

        int i, j, k;
        for (i = 0; i < r1; i++) {
            for (j = 0; j < c2; j++) {
                for (k = 0; k < c1; k++) {
                    if(k == 0) 
                      result(i, j) = 0.0f;
                    result(i, j) += mat1(i, k) * mat2(k, j);
                }
                result(i, j) *= alpha;
                result(i, j) += beta * self(i, j);
            }
        }
*/

        //   End profiling
        bsg_cuda_print_stat_kernel_end();
    }
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)

}

