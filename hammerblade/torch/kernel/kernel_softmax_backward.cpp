#include <kernel_common.hpp>
#include <stdint.h>
#include <cmath>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  //====================================================================
  // LogSoftMax Backward kernel
  // 03/26/2020 Lin Cheng
  //====================================================================

  __attribute__ ((noinline))  int tensorlib_log_softmax_backward(
          bsg_tensor_t* grad_input_p,
          bsg_tensor_t* grad_p,
          bsg_tensor_t* output_p,
          int32_t*      dim_p) {

    BSGTensor<float> gI(grad_input_p);
    BSGTensor<float> grad(grad_p);
    BSGTensor<float> output(output_p);
    int32_t dim = *dim_p;

    int32_t outer_size = 1;
    int32_t dim_size = grad.dim(dim);
    int32_t inner_size = 1;
    for (int32_t i = 0; i < dim; ++i)
      outer_size *= grad.dim(i);
    for (int32_t i = dim + 1; i < grad.ndim(); ++i)
      inner_size *= grad.dim(i);
    int32_t dim_stride = inner_size;
    int32_t outer_stride = dim_size * dim_stride;

    float* gradInput_data_base = (float*)grad_input_p->data;
    float* gradOutput_data_base = (float*)grad_p->data;
    float* output_data_base = (float*)output_p->data;

    brg_tile_for(outer_size * inner_size,
        [&](size_t i) {

          int32_t outer_idx = i / inner_size;
          int32_t inner_idx = i % inner_size;
          float* gradInput_data = gradInput_data_base + outer_idx * outer_stride + inner_idx;
          float* gradOutput_data = gradOutput_data_base + outer_idx * outer_stride + inner_idx;
          float* output_data = output_data_base + outer_idx * outer_stride + inner_idx;

          float sum = 0;
          for (size_t d = 0; d < dim_size; d++)
            sum += gradOutput_data[d * dim_stride];
          for (size_t d = 0; d < dim_size; d++)
            gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] -
              exp(output_data[d * dim_stride]) * sum;

        });
  }

    HB_EMUL_REG_KERNEL(tensorlib_log_softmax_backward,
        bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*,
        int32_t*);

}
