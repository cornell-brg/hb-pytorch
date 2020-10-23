//====================================================================
// Embedding merged with sum(1) kernel
// 10/23/2020 Lin Cheng
//====================================================================

#define BUF_SIZE 800
#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_embedding_with_sum(
          hb_tensor_t* sum_p,
          hb_tensor_t* weight_p,
          hb_tensor_t* indices_p,
          int32_t* padding_idx_p) {

    HBTensor<float>   sum(sum_p);
    HBTensor<float>   weight(weight_p);
    HBTensor<int64_t> indices(indices_p);
    const int32_t padding_idx = *padding_idx_p;
    const int32_t numel = weight.dim(1);
    const int32_t indices_numel = indices.numel();
    const int32_t batch_size = indices.dim(0);
    const int32_t embeddings_per_batch = indices.dim(1);

    std::cout << "padding_idx = " << padding_idx << " numel = " << numel << " indices_numel = " << indices_numel << " batch_size = " << batch_size << " embeddings_per_batch = " << embeddings_per_batch << std::endl;

    float acc[BUF_SIZE];
    // numel = P
    // indices_numel = N

    bsg_attr_remote float* sum_ptr = (bsg_attr_remote float*)sum.data_ptr();
    bsg_attr_remote float* weight_ptr = (bsg_attr_remote float*)weight.data_ptr();
    bsg_attr_remote int64_t* indices_ptr = (bsg_attr_remote int64_t*)indices.data_ptr();

    const uint32_t     sum_s0 =     sum.get_strides()[0];
    const uint32_t indices_s0 = indices.get_strides()[0];
    const uint32_t  weight_s0 =  weight.get_strides()[0];

    bsg_cuda_print_stat_kernel_start();

    for (size_t idx = bsg_id; idx < batch_size; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < batch_size) {
        // reset buffer
        bsg_unroll(16)
        for (size_t i = 0; i < numel; i++) {
          acc[i] = 0;
        }
        for (size_t emb = 0; emb < embeddings_per_batch; emb++) {
          const int32_t offset = (int32_t)*(indices_ptr + idx * indices_s0 + emb);
          if (offset != padding_idx) {
            bsg_attr_remote float* src = weight_ptr + offset * weight_s0;
            bsg_unroll(16)
            for (size_t i = 0; i < numel; i++) {
              acc[i] += *src;
              src++;
            }
          }
        }
        // write back
        bsg_attr_remote float* dst = sum_ptr + idx * sum_s0;
        bsg_unroll(16)
        for (size_t i = 0; i < numel; i++) {
          dst[i] = acc[i];
        }
      }
    }

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_embedding_with_sum, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, int32_t*);
}
