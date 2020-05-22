#include <kernel_common.hpp>

// common barrier for all kernels
#ifdef HB_EMUL
bsg_barrier g_barrier;
#endif // HB_EMUL
