#include <kernel_common.hpp>

// common reduction buffer
void* g_reduction_buffer;

// common barrier for all kernels
#ifdef HB_EMUL
bsg_barrier g_barrier;
#else
bsg_barrier<bsg_tiles_X, bsg_tiles_Y> g_barrier;
#endif // HB_EMUL


extern "C" {

  __attribute__ ((noinline))  int tensorlib_hb_startup(uint32_t* buffer) {

    buffer[2*__bsg_id] = 0;
    buffer[2*__bsg_id+1] = 0;
    g_reduction_buffer = (void*)buffer;
    g_barrier.reset();

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_hb_startup, uint32_t*)

}
