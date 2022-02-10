//====================================================================
// Arange kernel (only int is supported)
// Feb/23/2022 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp> 

extern "C" {

  __attribute__ ((noinline))  int tensorlib_arange(
          hb_tensor_t* _result,
          int32_t* _start,
          int32_t* _step,
          int32_t* _size) {
  
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
 
    int32_t* r = (int32_t*)HBTensor<int>(_result).data_ptr();
    int32_t start = *(_start);
    int32_t step = *(_step);
    int32_t size = *(_size);
    
    hb_tiled_for(size, [&](size_t i) {
      r[i] = start + i * step;
    });

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_arange, hb_tensor_t*, int32_t*, int32_t*, int32_t*)
}
