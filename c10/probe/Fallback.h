#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

class Fallback {
public:
  Fallback() : enabled(false) {}
  ~Fallback() = default;
  void enable() {enabled = true;}
  void disable() {enabled = false;}
  bool is_enabled() {return enabled;}
private:
  bool enabled;
};

C10_PROBE_API void fallback_enable();
C10_PROBE_API void fallback_disable();
C10_PROBE_API bool fallback_is_enabled();

extern Fallback g_fallback;

}} // namespace c10::probe
