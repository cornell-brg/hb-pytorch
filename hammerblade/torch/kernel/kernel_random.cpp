//====================================================================
// random kernel
// 23/01/2021 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <random>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_random_Int_(
          hb_tensor_t* _self,
          hb_tensor_t* _seed) {

    uint32_t seed = *(uint32_t*)((intptr_t)_seed->data);
    auto self = HBTensor<int32_t>(_self);
    std::default_random_engine generator;
    generator.seed(seed + __bsg_id);
    bsg_cuda_print_stat_kernel_start();
    hb_tiled_for(self.numel(), [&](size_t i) {
      self(i) = (int32_t)(generator() % (INT32_MAX + 1UL));
    });

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }
  
  HB_EMUL_REG_KERNEL(tensorlib_random_Int_, hb_tensor_t*, hb_tensor_t*)
}
    
