//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_add(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    // Get strides
    uint32_t _c_strides = *(uint32_t*)((intptr_t)res->strides);
    uint32_t _a_strides = *(uint32_t*)((intptr_t)a->strides);
    uint32_t _b_strides = *(uint32_t*)((intptr_t)b->strides);
    // Get starting elements of tensor data
    intptr_t _c = (intptr_t)res->data;
    intptr_t _a = (intptr_t)a->data;
    intptr_t _b = (intptr_t)b->data;
    float _alpha = *alpha;
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise add
    for (int i = start; i < end; i++) {
        *(float*)(_c) = *(float*)(_a) + (_alpha * *(float*)(_b));
        _c += _c_strides;
        _a += _a_strides;
        _b += _b_strides;
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_add, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)

}
