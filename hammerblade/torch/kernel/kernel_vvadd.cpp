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

    if (__bsg_id == 0) {
      for (size_t i = 0; i < result_p->N; i++){
        ((float*)((intptr_t) result_p->data))[i] = ((float*)((intptr_t)self_p->data))[i] + 
          ((float*)((intptr_t)other_p->data))[i];
      }
    }
    return 0;
  }
  
  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vvadd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
