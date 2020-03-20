//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

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
    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);

    int i, j, k;
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            result(i, j) = beta * self(i, j);
            for (k = 0; k < c1; k++) {
                result(i, j) += mat1(i, k) * mat2(k, j);
            }
            result(i, j) *= alpha;
        }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addmm, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*, float*)

}

