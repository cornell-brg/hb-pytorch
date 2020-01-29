#include <torch/hammerblade.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace hammerblade {
size_t device_count() {
  return at::detail::getHammerBladeHooks().getNumHBDevices();
}

bool is_available() {
  /*
   * Original CUDA comments:
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  // This function matches the semantics of at::cuda::is_available()
   */
  return hammerblade::device_count() > 0;
}
} // namespace hammerblade
} // namespace torch
