//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

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

    //-------------------------------------------------------
    // 2D result partitioning among Manycore tiles
    //
    // - Each tile computes a 2D slice of the results
    // - `k` dimesnion being in the outer loop is crucial,
    //   to make the maximum use of each element loaded from
    //   memory.
    //-------------------------------------------------------
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

