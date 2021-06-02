//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vvadd(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p,
          hb_tensor_t* other_p) {

    // Tutorial TODO:
    // Convert all low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);

    // Start profiling

    // Use a single tile only
    if (__bsg_id == 0) {
      // Tutorial TODO:
      // add elements from self and other together -- put the result in result
    }

    //   End profiling

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vvadd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
