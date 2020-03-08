//====================================================================
// Convolution kernel
// 03/08/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_convolution_forward(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          bsg_tensor_t* weight,
          bsg_vector_t* padding,
          bsg_vector_t* strides) {
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_convolution_forward,
     bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*,
     bsg_vector_t*, bsg_vector_t*);

}
