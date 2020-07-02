//====================================================================
// threshold kernel
// 03/19/2020 Lin Cheng (lc873@cornell.edu)
// 03/29/2020 Angela Zou (az292@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_threshold(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          float* _threshold_scalar_p,
          float* _value_scalar_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);
    float threshold = *_threshold_scalar_p;
    float value    = *_value_scalar_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [threshold, value](float self, float other) {
        if (self <= threshold) {
          return value;
        } else {
          return other;
        }
       },
       c, a, b);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_threshold, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, float*, float*)

}
