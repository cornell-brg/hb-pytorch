//====================================================================
// MSE backward kernel
// 08/18/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mse_backward(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          hb_tensor_t* t3_p,
          float* value_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    auto target = HBTensor<float>(t2_p);
    auto grad_output = HBTensor<float>(t3_p);
    float val = *value_p;

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [val](float self_val, float t1_val, float t2_val) {
        return val * (self_val - t1_val) * t2_val;
      },
      res, input, target, grad_output);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mse_backward, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, float*)

}
