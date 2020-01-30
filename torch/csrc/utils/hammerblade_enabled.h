#pragma once

namespace torch {
namespace utils {

static inline bool hammerblade_enabled() {
#ifdef USE_HB
  return true;
#else
  return false;
#endif
}

} // namespace utils
} // namespace torch
