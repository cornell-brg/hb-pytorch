#pragma once

/* This header hosts static variables that are essential to manage
 * HammerBlade devices
 * Feb 3, 2020
 * Lin Cheng
 */

#include <bsg_manycore_cuda.h>


namespace c10 {
namespace hammerblade {

extern hb_mc_dimension_t _hb_tg_dim;
extern hb_mc_dimension_t _hb_grid_dim;
extern hb_mc_device_t _hb_device;

#define PATH(x) pstr(x)
#define pstr(x) #x
static char _bin_path[] = PATH(HB_KERNEL_PATH); // path to device kernel binary
#undef pstr
#undef PATH

}} // namespace c10::hammerblade
