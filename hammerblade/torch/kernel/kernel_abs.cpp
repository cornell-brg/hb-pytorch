//====================================================================
// Element-wise abs kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_abs(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          float* value) {
    // Convert uint32_t pointers to correct types
    float*    _c = (float*)((intptr_t)res->data);
    float*    _a = (float*)((intptr_t)a->data);
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise abs
    for (int i = start; i < end; i++) {
        if (_a[i] < 0) {
            _c[i] = 0 - _a[i];
        } else {
            _c[i] = _a[i];
        }
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_abs, bsg_tensor_t*, bsg_tensor_t*, float*)

}
