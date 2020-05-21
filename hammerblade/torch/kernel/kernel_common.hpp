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
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#endif
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "hb_tensor.hpp"
#include <hb_assert.hpp>
#include <hb_parallel_for.hpp>
#include <hb_hw_patch.hpp>

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

#endif
