//====================================================================
// Sigmoid kernel
// 03/17/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sigmoid(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_element_wise_for(t0_p, t1_p, [&](float input) {
      return (1.0f / (1.0f + std::exp((-input))));
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sigmoid, bsg_tensor_t*, bsg_tensor_t*)

}

