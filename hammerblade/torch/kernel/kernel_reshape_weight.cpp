//=======================================================================
// Dense tensor add sparse tensor kernel
// 05/08/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=======================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_reshape_weight(
          hb_tensor_t* _new_indices,
          hb_tensor_t* _weight_indices,
          uint32_t* _dim2,
          uint32_t* _dim3,
          uint32_t* _nnz) {

    auto new_indices = HBTensor<int>(_new_indices);
    auto weight_indices = HBTensor<int>(_weight_indices);
    uint32_t dim2 = *_dim2;
    uint32_t dim3 = *_dim3;
    uint32_t nnz = *_nnz;
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = nnz;

    bsg_cuda_print_stat_kernel_start();

    for (size_t i = start; i < end; i = i + thread_num) {
      new_indices(i) = weight_indices(1, i) * dim2 * dim3 + weight_indices(2, i) * dim3 + weight_indices(3, i);
    }

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();  
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_reshape_weight, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*, uint32_t*);
}


