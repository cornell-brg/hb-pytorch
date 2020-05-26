//====================================================================
//// tanh kernel
//// 05/07/2020 Jack Weber (jlw422@cornell.edu)
////====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_tanh(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    hb_parallel_foreach(res, input,
      [&](float a) {
        return tanh(a);
    });
    //end profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_tanh, hb_tensor_t*, hb_tensor_t*)

}
