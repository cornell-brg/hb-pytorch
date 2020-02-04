#include "HammerBladeFunctions.h"

#include <mutex>

namespace c10 {
namespace hammerblade {

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  C10_HB_CHECK(hb_mc_device_init(&_hb_device, "HB_PYTORCH_PORT", 0));

  // XXX: apparently you need to load a binary file to setup allocator ...
  char bin_path[] = "/work/global/lc873/work/sdh/cosim/"
    "bsg_bladerunner/bsg_manycore/software/torch/add/add.riscv";
  C10_HB_CHECK(hb_mc_device_program_init(&_hb_device, bin_path, "default_allocator", 0));

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

 /* ------------------------------------------------------------------------------------
 * Interface to bsg_manycore runtime
 * -------------------------------------------------------------------------------------*/

int memcpy_host_to_device(void *dst, const void *src, uint32_t nbytes) {
  return hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_DEVICE);
}

int memcpy_device_to_host(void *dst, const void *src, uint32_t nbytes) {
  return hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_HOST);
}

}} // namespace c10::hammerblade
