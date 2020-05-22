//====================================================================
// On device memcpy kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_memcpy(
      void* dest,
      const void* src,
      uint32_t* n) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Perform memcpy if __bsg_id is 0
    if(__bsg_id == 0) {
      memcpy(dest, src, *n);
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_memcpy, void*, const void*, uint32_t*)

}
