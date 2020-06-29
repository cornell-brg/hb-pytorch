//====================================================================
// Sampled dense-dense matrix multiply
// 05/03/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_round(
          hb_tensor_t* a_p, //sample (sparse)
          hb_tensor_t* b_p, 
          hb_tensor_t* c_p, 
          hb_tensor_t* out_p) { //destination
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

  auto a = HBTensor<float>(a_p);
  auto b = HBTensor<float>(b_p);
  auto c = HBTensor<float>(c_p);
  auto res = HBTensor<float>(out_p);

    hb_tiled_foreach(inp, res,
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
