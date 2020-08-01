//=======================================================================
// Dense tensor add sparse tensor kernel
// 03/22/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=======================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_dense_sparse_add(
          hb_tensor_t* _result,
          hb_tensor_t* _dense,
          hb_tensor_t* _indices,
          hb_tensor_t* _values,
          float* _alpha) {

    auto result = HBTensor<float>(_result);
    auto dense = HBTensor<float>(_dense);
    auto indices = HBTensor<int>(_indices);
    auto values = HBTensor<float>(_values);
    float alpha= *_alpha; 
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = values.numel();

    bsg_cuda_print_stat_kernel_start();

    for (uint32_t i = start; i < end; i = i + thread_num) {
      uint32_t offset = 0;
      uint32_t coo = i;
      for(uint32_t d = 0; d < dense.ndim(); d++) {
        offset = offset + indices(coo) * dense.stride(d);
        coo = coo + indices.stride(0);
      }
      result(offset) = result(offset) + alpha * values(i);
    }

    bsg_cuda_print_stat_kernel_end(); 
    g_barrier.sync();  
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dense_sparse_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)
}


