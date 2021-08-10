//====================================================================
// MSE kernel
// 08/10/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mse(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {

    auto a = HBTensor<float>(t0_p);
    auto b = HBTensor<float>(t1_p);
    auto res = HBTensor<float>(t2_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](float a, float b){
        a = a - b;
        a = a * a;
        return a;
      },
      a, b, res);

    // End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mse, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

