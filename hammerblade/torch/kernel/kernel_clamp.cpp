//====================================================================
// Clamp kernels
// 04/23/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_clamp(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          float* min_p,
          float* max_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    float min = *min_p;
    float max = *max_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [&](float a) {
        return a < min ? min : (a > max ? max : a);
      },
      res, input);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_clamp_min(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          float* min_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    float min = *min_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [&](float a) {
        return a < min ? min : a;
      },
      res, input);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_clamp_max(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          float* max_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    float max = *max_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [&](float a) {
        return a > max ? max : a;
      },
      res, input);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_clamp, hb_tensor_t*, hb_tensor_t*, float*, float*)
  HB_EMUL_REG_KERNEL(tensorlib_clamp_min, hb_tensor_t*, hb_tensor_t*, float*)
  HB_EMUL_REG_KERNEL(tensorlib_clamp_max, hb_tensor_t*, hb_tensor_t*, float*)

}
