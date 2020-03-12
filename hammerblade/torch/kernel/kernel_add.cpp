//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <brg_element_for.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_add(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    float _alpha = *alpha;
    auto _c = BRGIteratorTensor<float*>(res);
    auto _a = BRGIteratorTensor<float*>(a);
    auto _b = BRGIteratorTensor<float*>(b);
    // Calculate elements per tile
    uint32_t len_per_tile = res->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t   end = start + len_per_tile;
    end = (end > res->N)  ? res->N : end;
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    // Element-wise add
    for (int i = start; i < end; i++) {
        *(*_c) = *(*_a) + _alpha * (*(*_b));
        _c++;
        _a++;
        _b++;
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_add, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, float*)

}
