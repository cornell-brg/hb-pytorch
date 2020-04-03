//====================================================================
// Mean reduction kernel
// 04/03/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling


extern "C" {

  __attribute__ ((noinline))  int tensorlib_mean(
          bsg_tensor_t* out_,
          bsg_tensor_t* in_,
          uint32_t* num_reduction_dim_p) {
    auto out = BSGTensor<float>(out_);
    auto in = BSGTensor<float>(in_);
    uint32_t num_reduction_dim = *num_reduction_dim_p;
    auto ndim = in.ndim();

    bsg_cuda_print_stat_kernel_start();

    // there could be more than 1 dims
    uint32_t elements_to_collect = in.dim(0);
    for(auto i = 1; i < num_reduction_dim; i++) {
      elements_to_collect *= in.dim(i);
    }

    auto reduce = [](float& partial_result, float input) {
                    partial_result += input;
                  };

    auto project = [&](float result) {
                    return result / elements_to_collect;
                   };

    binary_reduction(out, in, ndim, num_reduction_dim,
          elements_to_collect, reduce, project);

    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mean, bsg_tensor_t*, bsg_tensor_t*, uint32_t*)


  __attribute__ ((noinline))  int tensorlib_mean_simple(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p) {
    uint32_t input_stride = *(uint32_t*)((intptr_t)t1_p->strides);
    intptr_t output_base_p  = (intptr_t)t0_p->data;
    intptr_t input_base_p  = (intptr_t)t1_p->data;
    float sum_ = 0.0f;
    bsg_cuda_print_stat_kernel_start();
    if (__bsg_id == 0) {
      for(size_t i = 0; i < t1_p->N; i++) {
        sum_ += *(float*)(input_base_p);
        input_base_p += input_stride;
      }
      *(float*)(output_base_p) += (sum_ / t1_p->N);
    }
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mean_simple, bsg_tensor_t*, bsg_tensor_t*);

}
