//====================================================================
// Dot product kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_dot(
          bsg_tensor_t* _c,
          bsg_tensor_t* _a,
          bsg_tensor_t* _b) {
    auto c = BSGTensor<float>(_c);
    auto a = BSGTensor<float>(_a);
    auto b = BSGTensor<float>(_b);
    float sum = 0.0f;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Partial dot product sum
    brg_tile_for(a.numel(), [&](size_t i) {
        sum += a[{i}] * b[{i}];
    });
    // XXX: this operation should be atomic and consider the case in which
    // there are more than 1 tile
    c[{0}] = sum;
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dot, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*)

}
