#pragma once

#include <c10/hammerblade/HammerBladeFunctions.h>

namespace at {
namespace hammerblade {

  inline Device getDeviceFromPtr(void* ptr) {
    return {DeviceType::HAMMERBLADE, static_cast<int16_t>(c10::hammerblade::current_device())};
  }

}} // namespace at::hammerblade
