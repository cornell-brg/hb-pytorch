#include <kernel_common.hpp>
#include <stdint.h>
#include <cmath>
#include <limits>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  //====================================================================
  // LogSoftmax kernel
  // 03/20/2020 Bandhav Veluri
  //
  // Since softmax is defined over a vector, computation can be split into
  // independent computations on individual vectors. And, though, output
  // dimensions have to match those of input, computation doesn't need to
  // explicitly index dimensions other than `dim`. All dimensions above and
  // below can be iterated flat. For example:
  //
  // log_softmax(2x3x3x4x3 tensor, dim=2)
  //
  // can be simplified to,
  //
  // log_softmax(6x3x12 tensor, dim=2)
  //
  // Similarly, any dimension input tensor can be reduced into a 3-d tensor
  // with an outer dimension, softmax dimension and inner dimension. In
  // the example above, length of outer dimension = 6, length of softmax
  // dimension = 3 and lenght of inner dimension = 12.
  //====================================================================

  __attribute__ ((noinline))  int tensorlib_log_softmax(
          hb_tensor_t* out_,
          hb_tensor_t* in_,
          int32_t* dim_) {
    float* in_data = (float*) ((intptr_t) in_->data);
    float* out_data = (float*) ((intptr_t) out_->data);
    uint32_t* in_strides = (uint32_t*) ((intptr_t) in_->strides);
    uint32_t* in_sizes = (uint32_t*) ((intptr_t) in_->sizes);
    int32_t dim = *dim_;

    // Number of vectors to be softmaxed over.
    uint32_t num_softmax_axes = in_->N / in_sizes[dim];

    // Outer dimension stride
    uint32_t outer_stride = in_sizes[dim] * in_strides[dim];

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_parallel_for(num_softmax_axes, [&](size_t n) {
        uint32_t outer_index = n / in_strides[dim];
        uint32_t inner_index = n % in_strides[dim];
        uint32_t offset = outer_index * outer_stride + inner_index;

        // ----------------------------------------------------
        // LogSoftMax numerically stable simplification
        //
        // log_softmax(xi)
        //    = log(exp(xi) / (sigma_i(exp(xi))))
        //    = log(exp(xi - xmax) / (sigma_i(exp(xi - xmax))))
        //    = (xi - xmax) - log(sigma_i(exp(xi - xmax)))
        // ----------------------------------------------------

        // Compute xmax
        float xmax = std::numeric_limits<float>::lowest();
        for(uint32_t i = 0; i < in_sizes[dim]; ++i) {
          uint32_t index = offset + i * in_strides[dim];
          if(xmax < in_data[index]) {
            xmax = in_data[index];
          }
        }

        // Compute sigma_i(exp(xi))
        float exp_sum = 0.0f;
        for(uint32_t i = 0; i < in_sizes[dim]; ++i) {
          uint32_t index = offset + i * in_strides[dim];
          exp_sum += exp(in_data[index] - xmax);
        }

        // Compute log_softmax
        for(uint32_t i = 0; i < in_sizes[dim]; ++i) {
          uint32_t index = offset + i * in_strides[dim];
          out_data[index] = in_data[index] - xmax - log(exp_sum);
        }
    });

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_log_softmax,
     hb_tensor_t*, hb_tensor_t*, int32_t*);

}
