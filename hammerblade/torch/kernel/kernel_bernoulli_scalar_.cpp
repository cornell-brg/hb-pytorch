//====================================================================
// bernoulli_scalar_ kernel
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <random>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_bernoulli_scalar_(
          bsg_tensor_t* _self,
          bsg_tensor_t* _seed,
          float* _p) {
    // Unwrap common seed
    uint32_t seed = *(uint32_t*)((intptr_t)_seed->data);
    float p = *_p;
    auto self = BSGTensor<float>(_self);
    // RNG
    std::default_random_engine generator;
    generator.seed(seed + __bsg_id);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // bernoulli
    hb_tile_for(self.numel(), [&](size_t i) {
        float rand = distribution(generator);
        if (rand > p) {
          // 0
          // see ./aten/src/ATen/native/Dropout.cpp:55:  noise.bernoulli_(1 - p);
          self(i) = 0.0f;
        } else {
          // 1
          self(i) = 1.0f;
        }
    });
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_bernoulli_scalar_, bsg_tensor_t*,
                     bsg_tensor_t*, float*)

}
