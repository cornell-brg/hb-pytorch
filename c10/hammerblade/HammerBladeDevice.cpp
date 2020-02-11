#include <c10/hammerblade/HammerBladeDevice.h>

namespace c10 {
namespace hammerblade {

hb_mc_device_t _hb_device;

hb_mc_dimension_t _hb_tg_dim = { .x = 2, .y = 2};
hb_mc_dimension_t _hb_grid_dim = { .x = 1, .y = 1};

}} // namespace c10::hammerblade
