#pragma once

// This header provides C++ wrappers around commonly used HB API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

/*
 * inlcude bsg_manycore.h here
 * #include <bsg_manycore.h>
 */

#include "HammerBladeFunctions.h"

#include <mutex>

namespace c10 {
namespace hammerblade {

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  //TODO: put real HammerBlade init code here
  AT_WARN("FAKE initialization of HammerBlade Device!");
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
