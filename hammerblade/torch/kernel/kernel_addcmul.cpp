//====================================================================
// Point-wise addcmul kernel
// 04/23/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_addcmul(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          hb_tensor_t* t3_p,
          float* value_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    auto tensor1 = HBTensor<float>(t2_p);
    auto tensor2 = HBTensor<float>(t3_p);
    float value = *value_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [&](float input_val, float tensor1_val, float tensor2_val) {
        return input_val + value * tensor1_val * tensor2_val;
      },
      res, input, tensor1, tensor2);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addcmul, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, float*)

}
