//====================================================================
// Element-wise round kernel
// 04/11/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_round(
          hb_tensor_t* t0_p, // source tensor
          hb_tensor_t* t1_p) { //destination
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    HBTensor<uint32_t> res = HBTensor<uint32_t>(t0_p);
    HBTensor<uint32_t> input = HBTensor<uint32_t>(t1_p);

    hb_parallel_foreach(res, input,
      [](float a) {
        return roundf(a);
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_round, hb_tensor_t*, hb_tensor_t*)

}
