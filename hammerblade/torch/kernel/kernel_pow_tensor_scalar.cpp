//====================================================================
// power kernel for tensor with scalar exponent
// 08/13/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_pow_tensor_scalar(
          hb_tensor_t* _base,
          hb_tensor_t* _result,
          float* _exp_scalar) {
    auto base = HBTensor<float>(_base);
    auto result = HBTensor<float>(_result);
    float exp = *_exp_scalar;


    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    // similar to aten/src/ATen/native/cpu/PowKernel.cpp
    if (exp == 0.5) {
      hb_tiled_foreach([](float b) {
        return sqrtf(b);
      }, base, result);
    } else if (exp == 2.0) {
      hb_tiled_foreach([](float b) {
        return b*b;
      }, base, result);
    } else if (exp == 3.0) {
      hb_tiled_foreach([](float b) {
        return b*b*b;
      }, base, result);
    } else if (exp == -0.5) {
      hb_tiled_foreach([](float b) {
        return 1.0 / sqrtf(b);
      }, base, result);
    } else if (exp == -1.0) {
      hb_tiled_foreach([](float b) {
        return 1.0 / b;
      }, base, result);
    } else if (exp == -2.0) {
      hb_tiled_foreach([](float b) {
        return 1.0 / (b*b);
      }, base, result);
    } else {
      hb_tiled_foreach([exp](float b) {
        return powf(b, exp);
      }, base, result);
    }

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_pow_tensor_scalar, hb_tensor_t*, hb_tensor_t*,
                     float*)

}
