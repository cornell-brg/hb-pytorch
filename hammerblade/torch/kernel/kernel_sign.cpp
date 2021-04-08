//====================================================================
// Sign kernel
// 01/23/2021 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sign(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    
    auto inp = HBTensor<float>(t0_p);
    auto res = HBTensor<float>(t1_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](float a){
        return (0 < a) - (a < 0);
      },
      inp, res);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_sign, hb_tensor_t*, hb_tensor_t*)

}
