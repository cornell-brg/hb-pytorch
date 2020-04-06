//====================================================================
// Element-wise abs kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_abs(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);

    bsg_cuda_print_stat_kernel_start();

    hb_parallel_foreach(res, input,
      [&](float a) {
        return abs(a);
    });

    bsg_cuda_print_stat_kernel_end();

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_abs, hb_tensor_t*, hb_tensor_t*)

}
