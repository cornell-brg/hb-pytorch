//====================================================================
// Element-wise less than kernel
// 11/15/2021 Dani Song (ds2288@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_ge_Float(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    HBTensor<bool> out(t0_p);
    HBTensor<float> input(t1_p);
    HBTensor<float> other(t2_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](float input, float other) {
        return input >= other;
      },
      out, input, other);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_ge_Int(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    HBTensor<bool> out(t0_p);
    HBTensor<int32_t> input(t1_p);
    HBTensor<int32_t> other(t2_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](int32_t input, int32_t other) {
        return input >= other;
      },
      out, input, other);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_ge_Float, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
  HB_EMUL_REG_KERNEL(tensorlib_ge_Int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
