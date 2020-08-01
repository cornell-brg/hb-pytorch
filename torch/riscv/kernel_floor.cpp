//====================================================================
// Floor kernel
// 4/10/2020 Michelle Chao (mc2244@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_floor(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_tiled_foreach(
      [](float a) {
        return floor(a);
      },
      res, input);
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_floor, hb_tensor_t*, hb_tensor_t*)

}
