//====================================================================
// Element-wise round kernel
// 04/11/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_round(
          bsg_tensor_t* t0_p, // source tensor
          bsg_tensor_t* t1_p) { //destination
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    brg_tile_elementwise_for(t0_p, t1_p,
      [&](float a) {
        return roundf(a);
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_round, bsg_tensor_t*, bsg_tensor_t*)

}
