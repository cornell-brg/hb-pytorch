#ifndef _HB_COMMON_HPP
#define _HB_COMMON_HPP

// Pointers to remote locations (non-scratchpad) could be qualified
// with __remote. Doing so would tell compiler to assign a different
// address space to the contents of those pointers and latencies of
// memory accesses would be considered as 20 cycles. For example,
// `__remote float* foo;` essentially declares foo as `__remote float*`
// type, and the compiler assumes loads from `foo` to take 20 cycles
// on average.
#ifdef __clang__
#define __remote __attribute__((address_space(1)))
#else
#define __remote
#endif

// This macro is to protect the code from uncertainity with
// restrict/__restrict/__restrict__. Apparently some Newlib
// headers define __restrict as nothing, but __restrict__
// seems to work. So, we can use NOALIAS as our main way to
// resolve pointer alaising and possibly in future we could
// have `#ifdef`s here to make sure we use the right one in
// each circumstance.
#define NOALIAS __restrict__

#define PRAGMA(x) _Pragma(#x)
#ifdef __clang__
#define UNROLL(n) PRAGMA(unroll n)
#else
#define UNROLL(n) PRAGMA(GCC unroll n)
#endif

// =============================================================
// Workarounds for HB HW Issues
//
// This implementes a set of workarounds for HW
// issues that might be discovered on ASIC. The plan is to
// reproduce the bug in cosimulation and use verilog asserts
// to root cause the line of kernel code triggering the bug.
// After figuring out the line of kernel code triggering the
// bug, the verilog error message can used to find the right
// software fix in this header, and use fix that to patch the
// kernel code.
//
// For example, if a bug is triggered by a WAW violation between:
//
//   sizes = (uint32_t*) ((intptr_t) t->sizes);
//     204: 00c52f03            lw  x30,12(x10)
//   .
//   .
// (and)
//   .
//   .
//   strides[1] = (input.get_strides())[0];
//     3ac: 00092f03            lw  x30,0(x18)
// 
// The error message by verilog assertion would start with:
// [ERROR][VCORE] STALL_FORCE_WB WAW HAZARD
//
// A possible workaround is to use the macro HB_FIX_WAW_HAZARD
// on `sizes`:
//
//   sizes = (uint32_t*) ((intptr_t) t->sizes);
//   HB_FIX_WAW_HAZARD(sizes);
//   .
//   .
//   strides[1] = (input.get_strides())[0];
// =============================================================
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

#endif // _HB_COMMON_HPP
