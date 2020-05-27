//====================================================================
// Embedding backward kernel
// 04/22/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_embedding_backward(
          hb_tensor_t* grad_weight_p,
          hb_tensor_t* grad_p,
          hb_tensor_t* index_p,
          int32_t* padding_idx_p,
          int32_t* num_weights_p,
          int32_t* numel_p) {

    HBTensor<float> grad_weight(grad_weight_p);
    HBTensor<float> grad(grad_p);
    HBTensor<int32_t> index(index_p);
    int32_t padding_idx = *padding_idx_p;
    int32_t num_weights = *num_weights_p;
    int32_t numel = *numel_p;
    float* grad_weight_data = (float*)grad_weight.data_ptr();
    float* grad_data = (float*)grad.data_ptr();
    int32_t* index_data = (int32_t*)index.data_ptr();

    bsg_cuda_print_stat_kernel_start();

    hb_parallel_for(index.numel(), [&](size_t i) {
      if (index_data[i] != padding_idx) {
        int32_t k = index_data[i];
        if (k >= 0 && k < num_weights) {
          float scale = 1.0;
          float* dst = grad_weight_data + k * grad_weight.get_strides()[0];
          float* src = grad_data + i * grad.get_strides()[0];
          for (size_t j=0; j<numel; j++) {
            *dst += *src * scale;
            dst++;
            src++;
          }
        }
      }
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_embedding_backward, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, int32_t*,
                     int32_t*, int32_t*)

}
