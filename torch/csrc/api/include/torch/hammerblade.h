#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>

namespace torch {
namespace hammerblade {
/// Returns the number of HammerBlade devices available.
size_t TORCH_API device_count();

/// Returns true if HammerBlade device is available.
bool TORCH_API is_available();

} // namespace hammerblade
} // namespace torch
