//====================================================================
// Softmax kernel
// 03/20/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_log_softmax(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          long long* dim) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    bsg_printf("Running log_softmax\n");

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_log_softmax,
     bsg_tensor_t*, bsg_tensor_t*, long long*);

}
