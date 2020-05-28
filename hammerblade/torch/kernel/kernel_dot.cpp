//====================================================================
// Dot product kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_dot(
          hb_tensor_t* _c,
          hb_tensor_t* _a,
          hb_tensor_t* _b) {
    auto c = HBTensor<float>(_c);
    auto a = HBTensor<float>(_a);
    auto b = HBTensor<float>(_b);
    float sum = 0.0f;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Partial dot product sum
    hb_tiled_for(a.numel(), [&](size_t i) {
        sum += a(i) * b(i);
    });
    float *buffer = (float*)g_reduction_buffer;
    buffer[__bsg_id] = sum;
    g_barrier.sync();

    if(__bsg_id == 0) {
      float result = 0.0f;
      for(size_t idx = 0; idx < bsg_tiles_X * bsg_tiles_Y; idx++) {
        result += buffer[idx];
      }
      c(0) = result;
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dot, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
