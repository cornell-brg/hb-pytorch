//====================================================================
// HammerBlade kernel emulation header
// 03/02/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
//
// In order to support function name to function pointer mapping, the
// kernel auhtor needs to register the kernel function
//
// kernel:
// int  __attribute__ ((noinline)) tensorlib_add(
//        hb_tensor_t* res,
//        hb_tensor_t* a,
//        hb_tensor_t* b,
//        float* alpha)
//
// registration:
// HB_EMUL_REG_KERNEL(tensorlib_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)
//
// We always include this header, even if we are compiling for cosim

#ifndef _HAMMERBLADE_EMUL_H
#define _HAMMERBLADE_EMUL_H

#ifdef HB_EMUL
#include <kernel_trampoline.h>
#else
// If HB_EMUL is not define, then we define HB_EMUL_REG_KERNEL to be empty
#undef  HB_EMUL_REG_KERNEL
#define HB_EMUL_REG_KERNEL(...)
#endif

#endif
