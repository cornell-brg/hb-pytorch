//====================================================================
// Element-wise max kernel
// 23/06/2020 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_max(
          hb_tensor_t* result,
          hb_tensor_t* self,
          hb_tensor_t* other) {

    auto c = HBTensor<float>(result);
    auto a = HBTensor<float>(self);
    auto b = HBTensor<float>(other);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(c, a, b,
      [&](float a, float b) {
        return (a > b) ? a : b;
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_max, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
