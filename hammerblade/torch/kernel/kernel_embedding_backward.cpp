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
    uint32_t indices_numel = index.numel();

    // numel = P
    // indices_numel = N
    // num_weights = K

    bsg_attr_remote int32_t* index_data = (bsg_attr_remote int32_t*)index.data_ptr();
    bsg_attr_remote float* grad_weight_data = (bsg_attr_remote float*)grad_weight.data_ptr();
    bsg_attr_remote float* grad_data = (bsg_attr_remote float*)grad.data_ptr();
    uint32_t grad_weight_s0 = grad_weight.get_strides()[0];
    uint32_t grad_s0 = grad.get_strides()[0];
    uint32_t grad_s1 = grad.get_strides()[1];

    bsg_cuda_print_stat_kernel_start();

    for (size_t idx = bsg_id; idx < indices_numel; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < indices_numel) {
        int32_t offset = *(index_data + idx);
        if (offset != padding_idx) {
          bsg_attr_remote float* dst = grad_weight_data + offset * grad_weight_s0;
          bsg_attr_remote float* src = grad_data + idx * grad_s0;
          bsg_unroll(16)
          for (size_t i = 0; i < numel; i++) {
            *dst += *src;
            dst++;
            src += grad_s1;
          }
        }
      }
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_embedding_backward, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, int32_t*,
                     int32_t*, int32_t*)

}
