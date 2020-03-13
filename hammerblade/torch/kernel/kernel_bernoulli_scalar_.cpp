//====================================================================
// bernoulli_scalar_ kernel
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <random>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_bernoulli_scalar_(
          bsg_tensor_t* self,
          bsg_tensor_t* seed,
          float* p) {
    // Convert uint32_t pointers to correct types
    float* _self = (float*)((intptr_t)self->data);
    uint32_t _seed = *(uint32_t*)((intptr_t)seed->data);
    float _p = *p;
    // Calculate elements per tile
    uint32_t len_per_tile = self->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > self->N)  ? self->N : end;
    // RNG
    std::default_random_engine generator;
    generator.seed(_seed + __bsg_id);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // bernoulli
    for (int i = start; i < end; i++) {
      float rand = distribution(generator);
      if (rand < _p) {
        // 0
        _self[i] = 0.0f;
      } else {
        // 1
        _self[i] = 1.0f;
      }
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_bernoulli_scalar_, bsg_tensor_t*,
                     bsg_tensor_t*, float*)

}
