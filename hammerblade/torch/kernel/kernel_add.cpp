//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

// Uses hb_tiled_foreach_unroll
// Optimum unrolling for 4x4 Bladerunner: 8
// Optimum unrolling for 8x16 Bladerunner: 6

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling


extern "C" {

  __attribute__ ((noinline))  int tensorlib_add(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          float* alpha_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);
    float alpha = *alpha_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach_unroll<1>(c, a, b,
      [&](float a, float b) {
        return a + alpha * b;
    });
    

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)

}
