import argparse
import os
import subprocess
from aten.src.ATen.code_template import CodeTemplate

# get target tile group dims from commandline

parser = argparse.ArgumentParser(description='Configurate HB tile group dimenions')
parser.add_argument('--tile-group-X', action="store", dest="dimX", type=int)
parser.add_argument('--tile-group-Y', action="store", dest="dimY", type=int)

args = parser.parse_args()

host_file   = "c10/hammerblade/HammerBladeDevice.cpp"
device_file = "hammerblade/torch/kernel/kernel_common.hpp"

host_template = CodeTemplate("""\
#include <c10/hammerblade/HammerBladeDevice.h>

namespace c10 {
namespace hammerblade {

hb_mc_device_t _hb_device;

hb_mc_dimension_t _hb_tg_dim = { .x = ${dimX}, .y = ${dimY}};
hb_mc_dimension_t _hb_grid_dim = { .x = 1, .y = 1};

bool hb_mc_should_trace = false;

}} // namespace c10::hammerblade
""")

device_template = CodeTemplate("""\
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
#else
#define BSG_TILE_GROUP_X_DIM ${dimX}
#define BSG_TILE_GROUP_Y_DIM ${dimY}
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
""")


host_config   =   host_template.substitute({"dimX":args.dimX, "dimY":args.dimY})
device_config = device_template.substitute({"dimX":args.dimX, "dimY":args.dimY})

with open(host_file, 'w') as outfile:
  outfile.write(host_config)

with open(device_file, 'w') as outfile:
  outfile.write(device_config)

print("HB has been config'ed to run with {0} x {1} tile group".format(args.dimX, args.dimY))
print()
print("building ...")
build_run = subprocess.Popen(["python", "setup.py", "install"], env=os.environ)
build_run.wait()
