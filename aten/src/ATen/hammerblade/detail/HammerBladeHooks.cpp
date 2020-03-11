#include <ATen/hammerblade/detail/HammerBladeHooks.h>

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/detail/HammerBladeHooksInterface.h>
#include <ATen/hammerblade/HammerBladeDevice.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <c10/util/Exception.h>

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace hammerblade {
namespace detail {

std::unique_ptr<THBState, void (*)(THBState*)> HammerBladeHooks::initHammerBlade() const {
  // calling device_count will trigger lazy initialization
  c10::hammerblade::device_count();
  return std::unique_ptr<THBState, void (*)(THBState*)>(
      nullptr, [](THBState* p) {
        // no-op
      });
}

Device HammerBladeHooks::getDeviceFromPtr(void* data) const {
  return at::hammerblade::getDeviceFromPtr(data);
}

bool HammerBladeHooks::hasHammerBlade() const {
  return at::hammerblade::is_available();
}

int64_t HammerBladeHooks::current_device() const {
  return c10::hammerblade::current_device();
}

std::string HammerBladeHooks::showConfig() const {
  std::ostringstream oss;
  oss << "HammerBlade COSIM\n";
  return oss.str();
}

int HammerBladeHooks::getNumHBDevices() const {
  return c10::hammerblade::device_count();
}

Generator* HammerBladeHooks::getDefaultHammerBladeGenerator(DeviceIndex device_index) const {
  return at::hammerblade::detail::getDefaultHammerBladeGenerator(device_index);
}

using at::HammerBladeHooksRegistry;
using at::RegistererHammerBladeHooksRegistry;

REGISTER_HAMMERBLADE_HOOKS(HammerBladeHooks);

} // namespace detail
} // namespace hammerblade
} // namespace at
