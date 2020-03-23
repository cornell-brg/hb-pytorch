//====================================================================
// Element-wise mul and div kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mul(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          bsg_tensor_t* t2_p,
          float* alpha_p) {
    float alpha = *alpha_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_elementwise_for(t0_p, t1_p, t2_p,
        [&](float a, float b) {
          return a * alpha * b;
        });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mul, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)


  __attribute__ ((noinline))  int tensorlib_div(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p,
          bsg_tensor_t* t2_p,
          float* alpha_p) {
    float alpha = *alpha_p;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_elementwise_for(t0_p, t1_p, t2_p,
      [&](float a, float b) {
        return a / (alpha * b);
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_div, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)

}
