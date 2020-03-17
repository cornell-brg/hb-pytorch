//====================================================================
// Threshold kernel
// 03/17/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_threshold(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          bsg_tensor_t* t2_p,
          float* _threshold_scalar_p,
          float* _value_scalar_p) {
    float threshold = *_threshold_scalar_p;
    float value = *_value_scalar_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_element_wise_for(t0_p, t1_p, t2_p,
      [&](float x, float other) {
        return  x <= threshold ? value : other;
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }
  
  HB_EMUL_REG_KERNEL(tensorlib_threshold, bsg_tensor_t*, bsg_tensor_t*,
                     bsg_tensor_t*, float*, float*)

}
