//====================================================================
// Element-wise mul and div kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <common.h>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mul(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    // Convert uint32_t pointers to correct types
    float*    _c = (float*)((intptr_t)res->data);
    float*    _a = (float*)((intptr_t)a->data);
    float*    _b = (float*)((intptr_t)b->data);
    float _alpha = *alpha;
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise mul
    for (int i = start; i < end; i++) {
        _c[i] = _a[i] * (_alpha * _b[i]);
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mul, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)


  __attribute__ ((noinline))  int tensorlib_div(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    // Convert uint32_t pointers to correct types
    float*    _c = (float*)((intptr_t)res->data);
    float*    _a = (float*)((intptr_t)a->data);
    float*    _b = (float*)((intptr_t)b->data);
    float _alpha = *alpha;
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise div
    for (int i = start; i < end; i++) {
        _c[i] = _a[i] / (_alpha * _b[i]);
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_div, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)

}
