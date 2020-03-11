//====================================================================
// Dot product kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_dot(
          bsg_tensor_t* res,
          bsg_tensor_t*   a,
          bsg_tensor_t*   b) {
    // Convert uint32_t pointers to correct types
    float* _c = (float*)((intptr_t)res->data);
    float* _a = (float*)((intptr_t)a->data);
    float* _b = (float*)((intptr_t)b->data);
    float sum = 0.0f;
    // Calculate elements per tile
    uint32_t len_per_tile = a->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > a->N)  ? a->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Partial dot product sum
    for (int i = start; i < end; i++) {
        sum += _a[i] * _b[i];
    }
    // XXX: this operation should be atomic and consider the case in which
    // there are more than 1 tile
    _c[0] = sum;
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dot, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*)

}
