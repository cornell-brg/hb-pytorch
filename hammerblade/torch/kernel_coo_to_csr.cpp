//=======================================================================
// Dense tensor add sparse tensor kernel
// 03/22/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=======================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_coo_to_csr(
          hb_tensor_t* _csr,
          hb_tensor_t* _rowindices,
          uint32_t* _dim,
          uint32_t* _nnz) {
 
//    uint32_t _r_strides = *(uint32_t*)((intptr_t)res->strides);
//    uint32_t _d_strides = *(uint32_t*)((intptr_t)d->strides);
//    uint32_t _i_strides = *(uint32_t*)((intptr_t)indices->strides);
//    uint32_t _v_strides = *(uint32_t*)((intptr_t)valuse->strides);

    auto csr = HBTensor<int>(_csr);
    auto rowindices = HBTensor<int>(_rowindices);
    uint32_t dim = *_dim;
    uint32_t nnz = *_nnz;
    
    size_t len_per_tile = rowindices.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > rowindices.numel())  ? rowindices.numel() : end;

    bsg_cuda_print_stat_kernel_start();

    int h, hp0, hp1;
    for (size_t i = start; i < end; i++) {
      hp0 = rowindices(i);
      hp1 = (i+1 == nnz) ? dim : rowindices(i+1);
      if(hp0 != hp1) for(h = hp0; h < hp1; h++) {
        csr(h+1) = i+1;
      }
    }

    bsg_cuda_print_stat_kernel_end();   
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_coo_to_csr, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*)
}


