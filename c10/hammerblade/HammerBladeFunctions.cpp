#include "HammerBladeFunctions.h"

#include <mutex>

namespace c10 {
namespace hammerblade {

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  int rc = hb_mc_device_init(&_hb_device, "HB_PYTORCH_PORT", 0);
  if (rc != HB_MC_SUCCESS) {
    AT_ERROR("HammerBlade: failed to initialize device!");
  }

  // XXX: apparently you need to load a binary file to setup allocator ...
  char bin_path[] = "/work/global/lc873/work/sdh/cosim/"
    "bsg_bladerunner/bsg_manycore/software/torch/add/add.riscv";
  rc = hb_mc_device_program_init(&_hb_device, bin_path, "default_allocator", 0);
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
