//====================================================================
// exp kernel
// 04/13/2020 Yuyi He (yh383@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_exp(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_parallel_foreach(res, input,
      [&](float a) {
        return exp(a);
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_exp, bsg_tensor_t*, bsg_tensor_t*)

}