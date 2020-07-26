#include <kernel_common.hpp>
#include <string.h>
#include <stdint.h>

// common reduction buffer
void* g_reduction_buffer;

// common barrier for all kernels
#ifdef HB_EMUL
bsg_barrier g_barrier;
#else
bsg_barrier<bsg_tiles_X, bsg_tiles_Y> g_barrier;
#endif // HB_EMUL

// This is just Newlib's memcpy with __remote annotations
__remote void* hb_memcpy(__remote void* NOALIAS aa,
               const __remote void* NOALIAS bb,
               size_t n) {
  #define unlikely(X) __builtin_expect (!!(X), 0)

  #define BODY(a, b, t) { \
    t tt = *b; \
    a++, b++; \
    *(a - 1) = tt; \
  }

  __remote char *a = (__remote char *)aa;
  const __remote char *b = (const __remote char *)bb;
  __remote char *end = a + n;
  uintptr_t msk = sizeof (long) - 1;
  if (unlikely ((((uintptr_t)a & msk) != ((uintptr_t)b & msk))
           || n < sizeof (long)))
    {
small:
      if (__builtin_expect (a < end, 1))
    while (a < end)
      BODY (a, b, char);
      return aa;
    }

  if (unlikely (((uintptr_t)a & msk) != 0))
    while ((uintptr_t)a & msk)
      BODY (a, b, char);

  __remote long *la = (__remote long *)a;
  const __remote long *lb = (const __remote long *)b;
  __remote long *lend = (__remote long *)((uintptr_t)end & ~msk);

  if (unlikely (la < (lend - 8)))
    {
      while (la < (lend - 8))
    {
      long b0 = *lb++;
      long b1 = *lb++;
      long b2 = *lb++;
      long b3 = *lb++;
      long b4 = *lb++;
      long b5 = *lb++;
      long b6 = *lb++;
      long b7 = *lb++;
      long b8 = *lb++;
      *la++ = b0;
      *la++ = b1;
      *la++ = b2;
      *la++ = b3;
      *la++ = b4;
      *la++ = b5;
      *la++ = b6;
      *la++ = b7;
      *la++ = b8;
    }
    }

  while (la < lend)
    BODY (la, lb, long);

  a = (__remote char *)la;
  b = (const __remote char *)lb;
  if (unlikely (a < end))
    goto small;
  return aa;
}

// Memcpy DRAM to scratchpad
void* hb_memcpy_to_dmem(void* NOALIAS aa,
                        const __remote void* NOALIAS bb,
                        size_t n) {
  #define unlikely(X) __builtin_expect (!!(X), 0)

  #define BODY(a, b, t) { \
    t tt = *b; \
    a++, b++; \
    *(a - 1) = tt; \
  }

  char *a = (char *)aa;
  const __remote char *b = (const __remote char *)bb;
  char *end = a + n;
  uintptr_t msk = sizeof (long) - 1;
  if (unlikely ((((uintptr_t)a & msk) != ((uintptr_t)b & msk))
           || n < sizeof (long)))
    {
small:
      if (__builtin_expect (a < end, 1))
    while (a < end)
      BODY (a, b, char);
      return aa;
    }

  if (unlikely (((uintptr_t)a & msk) != 0))
    while ((uintptr_t)a & msk)
      BODY (a, b, char);

  long *la = (long *)a;
  const __remote long *lb = (const __remote long *)b;
  long *lend = (long *)((uintptr_t)end & ~msk);

  if (unlikely (la < (lend - 8)))
    {
      while (la < (lend - 8))
    {
      long b0 = *lb++;
      long b1 = *lb++;
      long b2 = *lb++;
      long b3 = *lb++;
      long b4 = *lb++;
      long b5 = *lb++;
      long b6 = *lb++;
      long b7 = *lb++;
      long b8 = *lb++;
      *la++ = b0;
      *la++ = b1;
      *la++ = b2;
      *la++ = b3;
      *la++ = b4;
      *la++ = b5;
      *la++ = b6;
      *la++ = b7;
      *la++ = b8;
    }
    }

  while (la < lend)
    BODY (la, lb, long);

  a = (char *)la;
  b = (const __remote char *)lb;
  if (unlikely (a < end))
    goto small;
  return aa;
}

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
