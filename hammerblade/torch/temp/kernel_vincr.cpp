//====================================================================
// Vector increment kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vincr(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p) {

    // Convert low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
      // Add 1 to each element
      for (size_t i = 0; i < self.numel(); i++) {
        result(i) = self(i) + 1;
      }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vincr, hb_tensor_t*, hb_tensor_t*)

}
