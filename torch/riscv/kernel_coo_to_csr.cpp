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

    auto csr = HBTensor<int>(_csr);
    auto rowindices = HBTensor<int>(_rowindices);
    uint32_t dim = *_dim;
    uint32_t nnz = *_nnz;
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = nnz;

    bsg_cuda_print_stat_kernel_start();

    int h, hp0, hp1;
    for (size_t i = start; i < end; i = i + thread_num) {
      hp0 = rowindices(i);
      hp1 = (i+1 == nnz) ? dim : rowindices(i+1);
      if(hp0 != hp1) for(h = hp0; h < hp1; h++) {
        csr(h+1) = i+1;
      }
    }

    bsg_cuda_print_stat_kernel_end();   
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_coo_to_csr, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*)
}


