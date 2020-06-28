#ifndef _KERNEL_COMMON_H
#define _KERNEL_COMMON_H

#include <cstring>
#include <cstdint>
#include <math.h>

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included.
#ifdef HB_EMUL
#include <emul_hb_device.h>
#define BSG_TILE_GROUP_X_DIM emul_hb_mesh_dim.x
#define BSG_TILE_GROUP_Y_DIM emul_hb_mesh_dim.y
#else
#define BSG_TILE_GROUP_X_DIM bsg_global_X
#define BSG_TILE_GROUP_Y_DIM (bsg_global_Y - 1)
#endif // HB_EMUL
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
// imaginary __bsg_pod_id and BSG_POD_DIM
#define __bsg_pod_id 0
#define BSG_POD_DIM 1
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_tile_group_barrier.hpp"
#include "hb_tensor.hpp"
#include <hb_assert.hpp>
#include <hb_tiled_for.hpp>
#include <hb_hw_patch.hpp>

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

__remote void* hb_memcpy(__remote void* NOALIAS dest,
                         const __remote void* NOALIAS src,
                         size_t n);

//====================================================================
// HammerBlade kernel emulation
// 03/02/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
// When emulation layer is enabled, macro HB_EMUL is defined
// In such case, we need to include kernel.h from c10/hammerblade/emul
// and we have to define init_kernel_starters
//
// Note: when emulation layer is enabled, this file is included when
// building c10/hammerblade/emul

#include <hammerblade_emul.hpp>

extern void* g_reduction_buffer;
#ifdef HB_EMUL
extern bsg_barrier g_barrier;
#else
extern bsg_barrier<bsg_tiles_X, bsg_tiles_Y> g_barrier;
#endif // HB_EMUL

#endif // _KERNEL_COMMON_H
