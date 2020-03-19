//====================================================================
// addmm kernel
// 03/09/2020 YOUR NAME HERE Lin Cheng (YOU EMAIL HERE, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_threshold(
          bsg_tensor_t* res,
          bsg_tensor_t* self,
          bsg_tensor_t* other,
          float* threshold_scalar,
          float* value_scalar) {
    // TODO: Convert uint32_t pointers to correct types
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // TODO: Implement threshold
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }
  
  HB_EMUL_REG_KERNEL(tensorlib_threshold, bsg_tensor_t*, bsg_tensor_t*,
                     bsg_tensor_t*, float*, float*)

}
