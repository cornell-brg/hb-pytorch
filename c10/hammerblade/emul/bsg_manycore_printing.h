#ifndef _BSG_MANYCORE_PRINTING_H
#define _BSG_MANYCORE_PRINTING_H
#include <bsg_manycore_features.h>
#include <cstdint>
#include <cstddef>

// emulate bsg_time, which returns 0 only
inline uint64_t bsg_time() {
  return 0;
}

#endif
