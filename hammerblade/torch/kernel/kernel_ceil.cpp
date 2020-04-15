//====================================================================
// Ceil kernel
// 04/11/2020 Kofi Amoako Efah (kae87@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_ceil(
          bsg_tensor_t* t0_p,
          bsg_tensor_t* t1_p) {
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    brg_tile_elementwise_for(t0_p, t1_p,
      [&](float a) { //anonymous function for ceiling. Does the ceiling operation for every element in the tile.
        return ceil(a); // returns ceiling of argument.
    });
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_ceil, bsg_tensor_t*, bsg_tensor_t*)

}

