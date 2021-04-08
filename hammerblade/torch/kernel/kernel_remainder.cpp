//====================================================================
// Element-wise remainder kernel
// 20/01/2021 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_remainder_Int(
          hb_tensor_t* result,
          hb_tensor_t* self,
          hb_tensor_t* other) {

    auto c = HBTensor<int32_t>(result);
    auto a = HBTensor<int32_t>(self);
    auto b = HBTensor<int32_t>(other);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [&](int32_t a, int32_t b) {
        return a % b;
    },
    c, a, b);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_remainder_Int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
