#pragma once

#include <c10/core/TensorOptions.h>

// hammerblade_lazy_init() is always compiled, even for CPU-only builds.
// Thus, it does not live in the hammerblade/ folder.

namespace torch {
namespace utils {

// The INVARIANT is that this function MUST be called before you attempt
// to get a HammerBlade Type object from ATen.

void hammerblade_lazy_init();
void hammerblade_set_run_yet_variable_to_false();

static void maybe_initialize_hammerblade(const at::TensorOptions& options) {
  if (options.device().is_hammerblade()) {
    torch::utils::hammerblade_lazy_init();
  }
}

} // namespace utils
} // namespace torch
