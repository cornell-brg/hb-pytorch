#pragma once

/* This header hosts static variables that are essential to manage 
 * HammerBlade devices
 * Feb 3, 2020
 * Lin Cheng
 */

#include <c10/hammerblade/HammerBladeMacros.h>
#include <c10/core/Device.h>

#include <bsg_manycore_cuda.h>

#define PATH(x) pstr(x)
#define pstr(x) #x

namespace c10 {
namespace hammerblade {

extern hb_mc_dimension_t _hb_tg_dim;
extern hb_mc_dimension_t _hb_grid_dim;
extern hb_mc_device_t _hb_device;

static char _bin_path[] = PATH(HB_KERNEL_PATH); // path to device kernel binary

}} // namespace c10::hammerblade
