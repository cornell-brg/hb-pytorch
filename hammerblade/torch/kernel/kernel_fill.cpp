//====================================================================
// Value fill kernel
// 03/05/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_fill(
          bsg_tensor_t* res,
          float* value) {
    // Convert uint32_t pointers to correct types
    float*    _c = (float*)((intptr_t)res->data);
    float _value = *value;
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise add
    for (int i = start; i < end; i++) {
        _c[i] = _value;
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_fill, bsg_tensor_t*, float*)

}

