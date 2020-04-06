//====================================================================
// Value fill kernel
// 03/05/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_fill(
          bsg_tensor_t* t0_p,
          float* value_p) {
    auto res = BSGTensor<float>(t0_p);
    float value = *value_p;

    bsg_cuda_print_stat_kernel_start();

    hb_parallel_foreach(res, [&]() {
      return value;
    });

    bsg_cuda_print_stat_kernel_end();

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_fill, bsg_tensor_t*, float*)

}
