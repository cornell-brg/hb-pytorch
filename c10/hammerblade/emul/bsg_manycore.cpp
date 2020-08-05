#include <bsg_manycore.h>
#include <bsg_manycore_errno.h>

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

int hb_mc_manycore_trace_enable(hb_mc_manycore_t *mc) {
  return HB_MC_SUCCESS;
}

int hb_mc_manycore_trace_disable(hb_mc_manycore_t *mc) {
  return HB_MC_SUCCESS;
}

int hb_mc_manycore_log_enable(hb_mc_manycore_t *mc) {
  return HB_MC_SUCCESS;
}

int hb_mc_manycore_log_disable(hb_mc_manycore_t *mc) {
  return HB_MC_SUCCESS;
}

#ifdef __cplusplus
}
#endif

