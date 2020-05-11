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
 
//    uint32_t _r_strides = *(uint32_t*)((intptr_t)res->strides);
//    uint32_t _d_strides = *(uint32_t*)((intptr_t)d->strides);
//    uint32_t _i_strides = *(uint32_t*)((intptr_t)indices->strides);
//    uint32_t _v_strides = *(uint32_t*)((intptr_t)valuse->strides);

    auto new_indices = HBTensor<int>(_new_indices);
    auto weight_indices = HBTensor<int>(_weight_indices);
    uint32_t dim2 = *_dim2;
    uint32_t dim3 = *_dim3;
    uint32_t nnz = *_nnz;
    
    size_t len_per_tile = nnz / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > nnz)  ? nnz : end;

    bsg_cuda_print_stat_kernel_start();

    for (size_t i = start; i < end; i++) {
      new_indices(i) = weight_indices(1, i) * dim2 * dim3 + weight_indices(2, i) * dim3 + weight_indices(3, i);
    }

    bsg_cuda_print_stat_kernel_end();   
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_reshape_weight, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*, uint32_t*);
}


