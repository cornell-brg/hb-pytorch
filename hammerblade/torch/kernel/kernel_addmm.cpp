//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_DIM 8 // sqrt(4KB/4 byte/4 data matrix) = 15 max

extern "C" {

  // Handles the common case:
  // Assume dim_y and dim_x are both exactly BLOCK_SIZE
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

  // Handles the common case:
  // Assume dim_y and dim_x are both exactly BLOCK_SIZE
  void dram_to_sp_simple(
          float* dest,
          float coeff,
          HBTensor<float, 2> src,
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

  // Handles the general case:
  // dim_y and dim_x can be less than BLOCK_SIZE
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


  // Handles the general case:
  // dim_y and dim_x can be less than BLOCK_SIZE
  void dram_to_sp(
          float* dest,
          float coeff,
          HBTensor<float, 2> src,
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

    //-----------------
    // loop
    //----------------
    for(size_t k = 0; k < c1; k++) {
      hb_tiled_for(r1, c2, [&](size_t i, size_t j) {
          if(k == 0)
              result(i, j) = 0;

          result(i, j) += mat1(i, k) * mat2(k, j);
      });
    }

    hb_tiled_for(r1, c2, [&](size_t i, size_t j) {
        result(i, j) = beta * self(i, j) + alpha * result(i, j);
    });
    
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)

}

