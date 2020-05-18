//====================================================================
// Ceil kernel
// 5/17/2020 Kofi Efah (kae87)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_ceil(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_parallel_foreach(res, input,
      [&](float a) {
        return ceil(a); 
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_ceil, hb_tensor_t*, hb_tensor_t*)

}
