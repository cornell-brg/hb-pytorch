//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#define BLOCK_DIM 10 // sqrt(4KB/4 byte/4 data matrix) * (2/3) prevent loading too much due to offset

extern "C" {

  __attribute__ ((noinline))  int tensorlib_addmm(
          bsg_tensor_t* _result,
          bsg_tensor_t* _self,
          bsg_tensor_t* _mat1,
          bsg_tensor_t* _mat2,
          float* _beta,
          float* _alpha) {
    // TODO: Convert uint32_t pointers to correct types
    auto self = BSGTensor<float>(_self);
    auto mat1 = BSGTensor<float>(_mat1);
    auto mat2 = BSGTensor<float>(_mat2);
    auto result = BSGTensor<float>(_result);
    float beta = *_beta;
    float alpha = *_alpha;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // TODO: Implement addmm

    // v2: single tile, use blocking
    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);
    assert(c1 == r2);

    bsg_printf("r1 %d c1 %d r2 %d c2 %d\n", r1, c1, r2, c2);
    bsg_printf("BLOCK_DIM %d\n", BLOCK_DIM);

    // calculate number of row and col blocks in each matrix
    int m1_num_blk_per_row = std::ceil((float)r1/(float)BLOCK_DIM); // how many blocks in m1 per row
    int m1_num_blk_per_col = std::ceil((float)c1/(float)BLOCK_DIM); // how many blocks in m1 per col
    int m2_num_blk_per_row = std::ceil((float)r2/(float)BLOCK_DIM); // how many blocks in m2 per row
    int m2_num_blk_per_col = std::ceil((float)c2/(float)BLOCK_DIM); // how many blocks in m2 per col

    bsg_printf("m1_num_blk_per_row %d\n", m1_num_blk_per_row);
    bsg_printf("m1_num_blk_per_col %d\n", m1_num_blk_per_col);
    bsg_printf("m2_num_blk_per_row %d\n", m2_num_blk_per_row);
    bsg_printf("m2_num_blk_per_col %d\n", m2_num_blk_per_col);

    // calculate dimensions of the last row and col block in each matrix
    int m1_last_blk_dim_x = r1 % BLOCK_DIM; // x dimension of last block of mat1
    int m1_last_blk_dim_y = c1 % BLOCK_DIM; // y dimension of last block of mat1
    int m2_last_blk_dim_x = r2 % BLOCK_DIM; // x dimension of last block of mat2
    int m2_last_blk_dim_y = c2 % BLOCK_DIM; // y dimension of last block of mat2

    bsg_printf("m1_last_blk_dim_x %d\n", m1_last_blk_dim_x);
    bsg_printf("m1_last_blk_dim_y %d\n", m1_last_blk_dim_y);
    bsg_printf("m2_last_blk_dim_x %d\n", m2_last_blk_dim_x);
    bsg_printf("m2_last_blk_dim_y %d\n", m2_last_blk_dim_y);

    // iterate over result blocks
    for (int rr = 0; rr < m1_num_blk_per_row; rr++) { // rr is index of row block in result matrix
        for (int rc = 0; rc < m2_num_blk_per_col; rc++) { // rc is index of col block in result matrix
            // calculate current result block dimensions
            int res_dim_y = rr == m1_num_blk_per_row - 1 ? m1_last_blk_dim_y : BLOCK_DIM;
            int res_dim_x = rc == m2_num_blk_per_col - 1 ? m2_last_blk_dim_x : BLOCK_DIM;

            bsg_printf("-----------------rr %d rc %d\n", rr, rc);
            bsg_printf("res_dim_y %d res_dim_x %d\n", res_dim_y, res_dim_x);

            // initialize scratchpad result
            float sp_result[res_dim_y][res_dim_x] = {0};

            // load self into scratchpad
            float sp_self[res_dim_y][res_dim_x];
            for (int i = 0; i < res_dim_y; i++) {
                for (int j = 0; j < res_dim_x; j++) {
                    sp_self[i][j] = self(rr * res_dim_y + i, rc * res_dim_x + j);
                    bsg_printf("sp_self[i][j] %f\n", sp_self[i][j]);
                }
            }

            // process mat1 and mat2 for this result block
            // only care about blocks of mat1 in row rr
            // and blocks of mat2 in col rc
            for (int mat1x = 0, mat2y = 0; mat1x < m1_num_blk_per_col && mat2y < m2_num_blk_per_row; mat1x++, mat2y++) {
                assert(mat1x == mat2y);
                // calculate current block dimensions
                int mid_dim = mat1x == m1_num_blk_per_col - 1 ? m1_last_blk_dim_x : BLOCK_DIM;

                // load mat1 and mat2 into scratchpad
                float sp_mat1[res_dim_y][mid_dim];
                float sp_mat2[mid_dim][res_dim_x];

                for (int i = 0; i < res_dim_y; i++) {
                    for (int j = 0; j < mid_dim; j++) {
                        sp_mat1[i][j] = mat1(rr * res_dim_y + i, mat1x * mid_dim + j);
                        bsg_printf("sp_mat1[i][j] %f\n", sp_mat1[i][j]);
                    }
                }

                for (int i = 0; i < mid_dim; i++) {
                    for (int j = 0; j < res_dim_x; j++) {
                        sp_mat2[i][j] = mat2(mat2y * mid_dim + i, rc * res_dim_x + j);
                        bsg_printf("sp_mat2[i][j] %f\n", sp_mat2[i][j]);
                    }
                }

                // compute mat1 X mat2 for this block
                for (int i = 0; i < res_dim_y; i++) {
                    for (int j = 0; j < res_dim_x; j++) {
                        for (int k = 0; k < mid_dim; k++) {
                            sp_result[i][j] += sp_mat1[i][k] * sp_mat2[k][j];
                            bsg_printf("sp_result[i][j] %f\n", sp_result[i][j]);
                        }
                    }
                }


            }
            // result = beta * self + alpha * (mat1 X mat2)
            for (int i = 0; i < res_dim_y; i++) {
                for (int j = 0; j < res_dim_x; j++) {
                    sp_result[i][j] *= alpha;
                    sp_result[i][j] += beta * sp_self[i][j];
                    bsg_printf("final sp_result[i][j] %f\n", sp_result[i][j]);
                }
            }


            // copy this block back into DRAM
            for (int i = 0; i < res_dim_y; i++) {
                for (int j = 0; j < res_dim_x; j++) {
                    //result(rr * res_dim_y + i, rc * res_dim_x + j) = sp_result[i][j];
                    result(i, j) = sp_result[i][j];
                    bsg_printf("result(i,j) %f\n", result(i,j));
                }
            }


        }
    }

    // v1: naive version, single tile, access DRAM elm by elm
/*
    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);

    int i, j, k;
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            for (k = 0; k < c1; k++) {
                result(i, j) += mat1(i, k) * mat2(k, j);
            }
            result(i, j) *= alpha;
            result(i, j) += beta * self(i, j);
        }
    }
*/

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addmm, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*, float*)

}

