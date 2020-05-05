#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

class ExecutionRoute {
public:
  ExecutionRoute() : odometer(0) {}
  ~ExecutionRoute() = default;
  void reset();
  void add_waypoint(const std::string& kernel, bool redispatch);
  void print();
  bool should_redispatch(const std::string& kernel);
private:
  size_t odometer;
  std::vector<std::tuple<std::string, bool>> route;
  std::map<std::string, bool> beacons;
};

C10_PROBE_API void route_add_waypoint(const std::string& kernel, bool redispatch);
C10_PROBE_API bool should_redispatch(const std::string& kernel);

extern ExecutionRoute g_execution_route;

}} // namespace c10::probe
