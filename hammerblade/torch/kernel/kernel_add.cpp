//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_add(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          bsg_tensor_t* t2_p,
          float* alpha_p) {
    auto c = BSGTensor<float>(t0_p);
    auto a = BSGTensor<float>(t1_p);
    auto b = BSGTensor<float>(t2_p);
    float alpha = *alpha_p;

    bsg_cuda_print_stat_kernel_start();

    hb_parallel_foreach(c, a, b,
      [&](float a, float b) {
        return a + alpha * b;
    });

    bsg_cuda_print_stat_kernel_end();

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_add, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)

}
