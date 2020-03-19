//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_max_pool2d(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          bsg_tensor_t* indices,
          int* kH, int* kW,
          int* dH, int* dW,
          int* padH, int* padW,
          int* dilationH, int* dilationW) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    exit(1);

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_max_pool2d,
    bsg_tensor_t*,
    bsg_tensor_t*,
    bsg_tensor_t*,
    int*, int*,
    int*, int*,
    int*, int*,
    int*, int*);

}
