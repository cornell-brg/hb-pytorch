//====================================================================
// Bit-wise not int kernel
// 11/15/2021 Dani Song (ds2288@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_bitwise_not_bool(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    HBTensor<bool> out(t0_p);
    HBTensor<bool> input(t1_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](bool input) {
        return ~input;
      },
      out, input);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_bitwise_not_bool, hb_tensor_t*, hb_tensor_t*)

}
