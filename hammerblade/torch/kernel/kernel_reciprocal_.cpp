//====================================================================
// reciprocal_
// for each entry x in tensor_p, change it to 1/x
// 08/28/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {
  __attribute__ ((noinline))  int tensorlib_reciprocal_(
          hb_tensor_t* tensor_p
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    
    auto out = HBTensor<float>(tensor_p);
    auto numel = out.numel();

    hb_tiled_for(numel, [&](size_t i) {
      out(i) = 1.0 / out(i);
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_reciprocal_, hb_tensor_t*)

}
