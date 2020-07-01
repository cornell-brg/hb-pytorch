#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

class ExecutionRoute {
public:
  ExecutionRoute() : odometer(0), check_allclose(false) {}
  ~ExecutionRoute() = default;
  void reset();
  void add_waypoint(const std::string& kernel, bool redispatch);
  const std::string print();
  bool should_redispatch(const std::string& kernel);
  bool should_check_allclose() { return check_allclose; }
  void enable_allclose_check() { check_allclose = true; }
  void disable_allclose_check() { check_allclose = false; }
private:
  size_t odometer;
  std::vector<std::tuple<std::string, bool>> route;
  std::map<std::string, bool> beacons;
  bool check_allclose;
};

C10_PROBE_API bool route_add_waypoint(const std::string& kernel, bool redispatch);
C10_PROBE_API bool should_redispatch(const std::string& kernel);
C10_PROBE_API const std::string route_print();
C10_PROBE_API bool should_check_allclose();
C10_PROBE_API void enable_allclose_check();
C10_PROBE_API void disable_allclose_check();
C10_PROBE_API inline bool use_hb_redispatch() {
#ifdef HB_REDISPATCH
  return true;
#else
  return false;
#endif
}

extern ExecutionRoute g_execution_route;

}} // namespace c10::probe
