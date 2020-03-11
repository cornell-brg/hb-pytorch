#include <ATen/detail/HammerBladeHooksInterface.h>

#include <ATen/Generator.h>
#include <ATen/HammerBladeGenerator.h>
#include <c10/util/Optional.h>

namespace at {
namespace hammerblade {
namespace detail {

struct HammerBladeHooks : public at::HammerBladeHooksInterface {
  HammerBladeHooks(at::HammerBladeHooksArgs) {}
  std::unique_ptr<THBState, void (*)(THBState*)> initHammerBlade() const override;
  Device getDeviceFromPtr(void* data) const override;
  // std::unique_ptr<Generator> initHammerBladeGenerator(Context*) const override;
  Generator* getDefaultHammerBladeGenerator(DeviceIndex device_index = -1) const override;
  bool hasHammerBlade() const override;
  int64_t current_device() const override;
  std::string showConfig() const override;
  int getNumHBDevices() const override;
};

} // namespace at
} // namespace hammerblade
} // namespace detail
