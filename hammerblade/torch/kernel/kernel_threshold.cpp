//====================================================================
// threshold kernel
// 03/09/2020 Bandhav Veluri Lin Cheng (YOU EMAIL HERE, lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_threshold(
          bsg_tensor_t* res,
          bsg_tensor_t* self,
          bsg_tensor_t* other,
          float* threshold_scalar,
          float* value_scalar) {
    auto N = self->N; // number of elements
    auto out = (float*) ((intptr_t) res->data);
    auto in = (float*) ((intptr_t) self->data);
    auto threshold = *threshold_scalar;
    auto value = *value_scalar;

    // Calculate elements per tile
    uint32_t len_per_tile = N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t end = start + len_per_tile;
    end = (end > N)  ? N : end;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(int i = start; i < end; ++i) {
      if(in[i] > threshold) {
        out[i] = in[i];
      } else {
        out[i] = value;
      }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_threshold, bsg_tensor_t*, bsg_tensor_t*,
                     bsg_tensor_t*, float*, float*)

}
