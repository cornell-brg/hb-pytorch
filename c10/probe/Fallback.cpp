#include <c10/probe/Fallback.h>

namespace c10 {
namespace probe {

Fallback g_fallback;

// ========== Fallback C10_API ==========

void fallback_enable() {
  g_fallback.enable();
}

void fallback_disable() {
  g_fallback.disable();
}

bool fallback_is_enabled() {
  return g_fallback.is_enabled();
}

}} // namespace c10::probe
