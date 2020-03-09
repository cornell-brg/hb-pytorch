//====================================================================
// Convolution kernel
// 03/08/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include <iostream>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_convolution_forward(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          bsg_tensor_t* weight,
          bsg_vector_t* padding,
          bsg_vector_t* strides) {
    uint32_t n_pads = padding->N;
    uint32_t* _padding = (uint32_t*) ((intptr_t) padding->data);
    uint32_t n_strides = strides->N;
    uint32_t* _strides = (uint32_t*) ((intptr_t) strides->data);

    auto x = BSGTensor(input);
    std::cout << "input(0,1,1,1) = " << x(0, 1, 1, 1) << std::endl;

    auto p = BSGVector<uint32_t>(padding);
    std::cout << "padding[0] = " << p[0] << std::endl;

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_convolution_forward,
     bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*,
     bsg_vector_t*, bsg_vector_t*);

}
