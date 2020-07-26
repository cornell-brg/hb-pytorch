//====================================================================
// Ceil kernel
// 05/19/2020 Kofi Efah (kae87)
//====================================================================

// Uses hb_tiled_foreach_unroll with an unrolling factor of 6
// Tested to be optimum for 4x4 Bladerunner

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_ceil(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_tiled_foreach_unroll<6>(res, input,
      [&](float a) {
        return ceil(a);
      });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_ceil, hb_tensor_t*, hb_tensor_t*)

}
