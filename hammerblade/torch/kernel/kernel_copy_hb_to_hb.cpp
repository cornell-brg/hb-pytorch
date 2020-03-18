//====================================================================
// copy_hb_to_hb kernel
// 03/18/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================
// Can't call memcpy directly here. Since the src tensor's stride could
// be zero.

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_copy_hb_to_hb(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_element_wise_for(t0_p, t1_p,
      [&](float a) {
        return a;
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_hb_to_hb, bsg_tensor_t*, bsg_tensor_t*)

}
