#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>
#include <c10/hammerblade/HammerBladeFunctions.h>

namespace at {
namespace hammerblade {

inline bool is_available() {
  return c10::hammerblade::device_count() > 0;
}

CAFFE2_API Allocator* getHAMMERBLADEDeviceAllocator();

} // namespace hammerblade
} // namespace at
