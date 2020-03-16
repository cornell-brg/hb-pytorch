#ifndef _BSG_MANYCORE_H
#define _BSG_MANYCORE_H

#include <cassert>
#include <cstdio>
#include <cstdarg>

#ifdef __cplusplus
extern "C"{
#endif

int bsg_printf(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

static inline void bsg_print_int(int i) {
  fprintf(stderr, "%d\n", i);
}

static inline void bsg_print_unsigned(unsigned u) {
  fprintf(stderr, "%u\n", u);
}

static inline void bsg_print_hexadecimal(unsigned u) {
  fprintf(stderr, "%x\n", u);
}

static inline void bsg_print_float(float f) {
  fprintf(stderr, "%f\n", f);
}

static inline void bsg_print_float_scientific(float f) {
  fprintf(stderr, "%E\n", f);
}

static inline void bsg_cuda_print_stat_kernel_start() {
  return;
}

static inline void bsg_cuda_print_stat_kernel_end() {
  return;
}

#define bsg_fail() assert (1==0 /* bsg_fail is called */)

#endif // _BSG_MANYCORE_H
