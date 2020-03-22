#include <kernel_common.hpp>
#include <stdint.h>
#include <cmath>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  //====================================================================
  // LogSoftMax kernel
  // 03/20/2020 Bandhav Veluri
  //====================================================================

  __attribute__ ((noinline))  int tensorlib_log_softmax(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          int64_t* dim_) {
    float* in = (float*) ((intptr_t) input->data);
    float* out = (float*) ((intptr_t) output->data);
    uint32_t* in_strides = (uint32_t*) ((intptr_t) input->strides);
    int64_t dim = *dim_;
    uint32_t dimStride = in_strides[dim];

    // A group here is the set of all dimensions lower than `dim`. So, number
    // of elements per group, groupSize, is equal to the stride of the dimension
    // `dim - 1`.
    //
    // Granularity is equal to the size of the group. log_softmax is typically
    // called on the last dimension, in which case, granularity is pretty small
    // achieving fine-grained parallelism.
    uint32_t groupSize = (dim == 0) ? input->N : in_strides[dim-1];
    uint32_t numGroups = input->N / groupSize;
    uint32_t dimsPerGroup = groupSize / dimStride;

    // Calculate groups per tile
    uint32_t groups_per_tile = numGroups / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start_group = groups_per_tile * __bsg_id;
    uint32_t end_group = start_group + groups_per_tile;
    end_group = (end_group > numGroups)  ? numGroups : end_group;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(int group = start_group; group < end_group; ++group) {
      // Iterate over lower dimensions
      for(uint32_t ld = 0; ld < dimStride; ++ld) {
        uint32_t start_ind = group * groupSize + ld;

        /*****************************************************
         * LogSoftMax numerically stable simplification
         *
         * log_softmax(xi) = log(exp(xi) / (sigma_i(exp(xi))))
         *                 = xi - log(sigma_i(exp(xi)))
         *****************************************************/

        // Compute sigma_i(exp(xi))
        float exp_sum = 0.0f;
        for(uint32_t i = start_ind;
            i < (start_ind + dimsPerGroup * dimStride);
            i += dimStride) {
          exp_sum += exp(in[i]);
        }

        // Compute log_softmax
        for(uint32_t i = start_ind;
            i < (start_ind + dimsPerGroup * dimStride);
            i += dimStride) {
          out[i] = in[i] - log(exp_sum);
        }
      }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_log_softmax,
     bsg_tensor_t*, bsg_tensor_t*, int64_t*);

}
