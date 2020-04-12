#ifndef _HB_HW_PATCH_HPP
#define _HB_HW_PATCH_HPP

#ifndef HB_EMUL

// Fixes WAW violations in HW
//
// WAW violations are seen in cosimulation as errors starting with:
// [ERROR][VCORE] STALL_FORCE_WB WAW HAZARD
#define HB_FIX_WAW_HAZARD(var) \
  do {                         \
    asm volatile (             \
        "mv %0, %1;"           \
        : "=r" ((var))         \
        : "r" ((var))          \
        );                     \
  } while(0)

#else

#define HB_FIX_WAW_HAZARD(var)

#endif // ifndef HB_EMUL

#endif // _HB_HW_PATCH_HPP
