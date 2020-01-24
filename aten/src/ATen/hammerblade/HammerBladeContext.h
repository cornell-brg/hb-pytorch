#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>

namespace at {
namespace hammerblade {

CAFFE2_API Allocator* getHAMMERBLADEDeviceAllocator();

} // namespace hammerblade
} // namespace at
