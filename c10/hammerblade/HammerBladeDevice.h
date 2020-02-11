#pragma once

/* This header hosts static variables that are essential to manage 
 * HammerBlade devices
 * Feb 3, 2020
 * Lin Cheng
 */

#include <c10/hammerblade/HammerBladeMacros.h>
#include <c10/core/Device.h>

#include <bsg_manycore_cuda.h>

namespace c10 {
namespace hammerblade {

extern hb_mc_dimension_t _hb_tg_dim;
extern hb_mc_dimension_t _hb_grid_dim;
extern hb_mc_device_t _hb_device;

}} // namespace c10::hammerblade
