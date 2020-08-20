#ifndef _BSG_MANYCORE_H
#define _BSG_MANYCORE_H

#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <stdint.h>

#define __remote

#ifdef __cplusplus
extern "C"{
#endif

int bsg_printf(const char *fmt, ...);

typedef struct hb_mc_manycore {
        const char *name;      //!< the name of this manycore
        // hb_mc_config_t config; //!< configuration of the manycore
        void *platform;        //!< machine-specific data pointer
        int dram_enabled;      //!< operating in no-dram mode?
} hb_mc_manycore_t;

/**
 * Enable trace file generation (vanilla_operation_trace.csv)
 * @param[in] mc    A manycore instance initialized with hb_mc_manycore_init()
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_manycore_trace_enable(hb_mc_manycore_t *mc);

/**
 * Disable trace file generation (vanilla_operation_trace.csv)
 * @param[in] mc    A manycore instance initialized with hb_mc_manycore_init()
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_manycore_trace_disable(hb_mc_manycore_t *mc);

/**
 * Enable log file generation (vanilla.log)
 * @param[in] mc    A manycore instance initialized with hb_mc_manycore_init()
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_manycore_log_enable(hb_mc_manycore_t *mc);

/**
 * Disable log file generation (vanilla.log)
 * @param[in] mc    A manycore instance initialized with hb_mc_manycore_init()
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_manycore_log_disable(hb_mc_manycore_t *mc);


int hb_mc_manycore_get_cycle(hb_mc_manycore_t *mc, uint64_t *time);


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

static inline void bsg_cuda_print_stat_start(uint32_t tag) {
  return;
}

static inline void bsg_cuda_print_stat_end(uint32_t tag) {
  return;
}


#define bsg_fail() assert (1==0 /* bsg_fail is called */)

#define __remote

#define NOALIAS

#define UNROLL(n)

#endif // _BSG_MANYCORE_H
