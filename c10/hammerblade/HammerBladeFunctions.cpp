#include "HammerBladeFunctions.h"

#include <mutex>

namespace c10 {
namespace hammerblade {

static hb_mc_dimension_t _hb_tg_dim = { .x = 2, .y = 2};
static hb_mc_dimension_t _hb_grid_dim = { .x = 1, .y = 1};
static hb_mc_device_t _hb_device;

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  int rc = hb_mc_device_init(&_hb_device, "HB_PYTORCH_PORT", 0);
  if (rc != HB_MC_SUCCESS) {
    AT_ERROR("HammerBlade: failed to initialize device!");
  }
  AT_WARN("HammerBlade Device Initialized\n");
  return;
}
} // namespace unnamed

DeviceIndex device_count() noexcept {
  // Assuming that we always have 1 HammerBlade manycore device
  int count = 1;
  // Lazy inialization
  std::call_once(init_flag, initHammerBladeDevice);
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device = -1;
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  return; // no-op
}


}} // namespace c10::hammerblade
