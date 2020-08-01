//====================================================================
// Element-wise mul and div kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>


extern "C" {

  __attribute__ ((noinline))  int tensorlib_mul(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);

    bsg_cuda_print_stat_kernel_start();
    
    hb_tiled_foreach_unroll<1>(c, a, b,
        [&](float a, float b) {
          return a * b;
        });
    

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mul, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_div(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach_unroll<1>(c, a, b,
      [&](float a, float b) {
        return a / b;
      });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_div, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
