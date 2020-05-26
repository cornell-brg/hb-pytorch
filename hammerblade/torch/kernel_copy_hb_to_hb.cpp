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
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<uint32_t>(t0_p);
    auto input = HBTensor<uint32_t>(t1_p);

    bsg_cuda_print_stat_kernel_start();

    hb_parallel_foreach(res, input,
      [](uint32_t a) {
        return a;
    });

    bsg_cuda_print_stat_kernel_end();
    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_hb_to_hb, hb_tensor_t*, hb_tensor_t*)

}
