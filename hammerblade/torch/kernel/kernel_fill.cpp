//====================================================================
// Value fill kernel
// 03/05/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_fill(
          bsg_tensor_t* t0_p,
          float* value_p) {
    float value = *value_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // even though here a is *NOT* used, we need this
    // parameter for tpye inference
    brg_tile_element_wise_for(t0_p, [&](float a = 0.0f) {
      return value;
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_fill, bsg_tensor_t*, float*)

}
