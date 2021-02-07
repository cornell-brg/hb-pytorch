//====================================================================
// Sigmoid kernel
// 03/17/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sigmoid(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {

    auto inp = HBTensor<float>(t0_p);
    auto res = HBTensor<float>(t1_p);

      // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [](float a){
        a = expf(-a);
        a = 1 + a;
        a = 1/a;
        return a;
      },
      inp, res);

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sigmoid, hb_tensor_t*, hb_tensor_t*)

}

