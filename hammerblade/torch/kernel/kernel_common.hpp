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
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
//Macros for triggering saif generation
#define bsg_saif_start()
#define bsg_saif_end()
#else
#define BSG_TILE_GROUP_X_DIM bsg_global_X
#define BSG_TILE_GROUP_Y_DIM bsg_global_Y
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#endif // HB_EMUL
// imaginary __bsg_pod_id and BSG_POD_DIM
#define __bsg_pod_id 0
#define BSG_POD_DIM 1
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_tile_group_barrier.hpp"
#include "bsg_manycore_atomic.h"
#include "hb_tensor.hpp"
#include <hb_assert.hpp>
#include <hb_tiled_for.hpp>
#include <hb_blocked_for.hpp>
#include <hb_spatial_for.hpp>
#include <hb_common.hpp>

bsg_attr_remote void* hb_memcpy(bsg_attr_remote void* bsg_attr_noalias dest,
                         const bsg_attr_remote void* bsg_attr_noalias src,
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
