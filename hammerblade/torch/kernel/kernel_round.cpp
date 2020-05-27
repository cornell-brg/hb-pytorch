//====================================================================
// Element-wise round kernel
// 05/03/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_round(
          hb_tensor_t* t0_p, // input tensor
          hb_tensor_t* t1_p) { //destination
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

  auto inp = HBTensor<float>(t0_p);
  auto res = HBTensor<float>(t1_p);

    hb_parallel_foreach(inp, res,
      [&](float a) {
        return rintf(a);
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_round, hb_tensor_t*, hb_tensor_t*)

}
