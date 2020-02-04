#pragma once

#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>

#include <bsg_manycore_errno.h>

/* This header hosts error checking macros for bsg_manycore runtime
 * We trigger AT_ERROR when an error occurs.
 * Feb 3, 2020
 * Lin Cheng
 */

#define C10_HB_CHECK(EXPR)                                              \
  do {                                                                  \
    int __rc = EXPR;                                                    \
    if (__rc != HB_MC_SUCCESS) {                                        \
      TORCH_CHECK(false, "HammerBlade error: ", hb_mc_strerror(__rc));  \
    }                                                                   \
  } while (0)

