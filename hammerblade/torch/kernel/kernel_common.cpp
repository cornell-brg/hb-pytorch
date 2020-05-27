#include <kernel_common.hpp>

// common barrier for all kernels
#ifdef HB_EMUL
bsg_barrier g_barrier;
#else
bsg_barrier<bsg_tiles_X, bsg_tiles_Y> g_barrier;
#endif // HB_EMUL


extern "C" {

  __attribute__ ((noinline))  int tensorlib_barrier_workaround() {

    g_barrier.reset();

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_barrier_workaround)

}
