#include <ATen/hammerblade/HammerBladeContext.h>

/*
 * inlcude Fake HB allocator
 */
#include <c10/core/CPUAllocator.h>

namespace at {
namespace hammerblade {

Allocator* getHAMMERBLADEDeviceAllocator() {
  return at::GetAllocator(DeviceType::HAMMERBLADE);
}

} // namespace hammerblade
} // namespace at
