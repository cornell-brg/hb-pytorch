#include <bsg_manycore.h>

#ifdef __cplusplus
extern "C"{
#endif

int bsg_printf(const char *fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(stderr, fmt, argptr);
  va_end(argptr);
  return 0;
}

#ifdef __cplusplus
}
#endif

