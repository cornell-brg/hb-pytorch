//====================================================================
// Point-wise addcdiv kernel
// 04/23/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

// Uses hb_tiled_foreach_unroll with an unrolling factor of 6
// Tested to be optimum for 4x4 Bladerunner

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_addcdiv(
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

    hb_tiled_foreach_unroll<6>(res, input, tensor1, tensor2,
      [&](float input_val, float tensor1_val, float tensor2_val) {
        return input_val + value * tensor1_val / tensor2_val;
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_addcdiv, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, float*)

}
