#include <c10/probe/Route.h>

#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace c10 {
namespace probe {

ExecutionRoute g_execution_route;

// ========== ExecutionRoute Member ==========

void ExecutionRoute::reset() {
  route.clear();
  beacons.clear();
}

void ExecutionRoute::add_waypoint(const std::string& kernel, bool redispatch) {
  route.push_back(std::make_tuple(kernel, redispatch));
  beacons[kernel] = true;
}

void ExecutionRoute::print() {
  json chart_json = json::array();
  for (const auto& wp : route) {
    json kernel_json;
    kernel_json["signature"] = std::get<0>(wp);
    kernel_json["offload"] = std::get<1>(wp);
    chart_json.push_back(kernel_json);
  }
  std::cerr << chart_json.dump(4) << std::endl;
}

bool ExecutionRoute::should_redispatch(const std::string& kernel) {
  if (beacons.find(kernel) != beacons.end()) {
    std::cerr << "at top level kernel " << kernel << std::endl;
    std::cerr << "should I redispatch? 1/0" << std::endl;
    int res = 0;
    std::cin >> res;
    if (res != 0) {
      std::cerr << "redispatching..." << std::endl;
      return true;
    }
  }
  return false;
}

// ========== ExecutionRoute C10_API ==========

void route_add_waypoint(const std::string& kernel, bool redispatch) {
  g_execution_route.add_waypoint(kernel, redispatch);
  // hack
  g_execution_route.print();
}

bool should_redispatch(const std::string& kernel) {
  return g_execution_route.should_redispatch(kernel);
}

}} // namespace c10::probe
