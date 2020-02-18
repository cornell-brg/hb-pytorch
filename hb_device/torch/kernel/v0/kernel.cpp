#include <cstring>
#include <cstdint>
#include <math.h>

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_tile_group_barrier.h"
#include "bsg_tensor.hpp"

INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1,
    0, bsg_tiles_Y-1);

template 
<typename TA, typename TB, typename TC, typename TD, typename Func>
int __attribute__ ((noinline)) vector_op(TA *A, TB *B, TC *C, TD alpha,
    uint32_t N, Func op) {
  
  uint32_t len_per_tile = ceil((float) N / 
      (float) (bsg_tiles_X * bsg_tiles_Y));
  
	for (int i = 0; i < len_per_tile; ++i) {
    int index = len_per_tile * __bsg_id + i;

    if(index > N)
      break;

    C[index] = op(A[index], alpha * B[index]);
	}

  // Tile group size in PyTorch is set to 1,1 as of now.
  // bsg_tile_group_barrier(&r_barrier, &c_barrier);

  return 0;
}

/* We wrap all external-facing C++ kernels with `extern "C"` to
 * prevent name mangling
 */
extern "C" {

  int  __attribute__ ((noinline)) tensorlib_add(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    int rc;
    bsg_cuda_print_stat_kernel_start();
    rc = vector_op(a->data, b->data, res->data, *alpha, res->N,
        [](float a, float b) { return a + b; });
    bsg_cuda_print_stat_kernel_end();
    return rc;
  }

  int  __attribute__ ((noinline)) tensorlib_mul(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    int rc;
    bsg_cuda_print_stat_kernel_start();
    rc = vector_op(a->data, b->data, res->data, *alpha, res->N,
        [](float a, float b) { return a * b; });
    bsg_cuda_print_stat_kernel_end();
    return rc;
  }

  int  __attribute__ ((noinline)) tensorlib_div(
          bsg_tensor_t* res,
          bsg_tensor_t* a,
          bsg_tensor_t* b,
          float* alpha) {
    int rc;
    bsg_cuda_print_stat_kernel_start();
    rc = vector_op(a->data, b->data, res->data, *alpha, res->N,
        [](float a, float b) { return a / b; });
    bsg_cuda_print_stat_kernel_end();
    return rc;
  }

}
