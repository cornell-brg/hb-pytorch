//====================================================================
// Element-wise abs kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_abs(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          float* value_p) {
    // value is *NOT* used here.
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_elementwise_for(t0_p, t1_p,
      [&](float a) {
        return abs(a);
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_abs, bsg_tensor_t*, bsg_tensor_t*, float*)

}
