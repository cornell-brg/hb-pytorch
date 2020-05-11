#include <c10/probe/Route.h>
#include <c10/util/Exception.h>

#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace c10 {
namespace probe {

ExecutionRoute g_execution_route;

// ========== ExecutionRoute Member ==========

void ExecutionRoute::reset() {
  odometer = 0;
  route.clear();
  beacons.clear();
}

void ExecutionRoute::add_waypoint(const std::string& kernel, bool redispatch) {
  route.push_back(std::make_tuple(kernel, redispatch));
  beacons[kernel] = true;
}

const std::string ExecutionRoute::print() {
  std::stringstream buffer;
  json chart_json = json::array();
  for (const auto& wp : route) {
    json kernel_json;
    kernel_json["signature"] = std::get<0>(wp);
    kernel_json["offload"] = std::get<1>(wp);
    chart_json.push_back(kernel_json);
  }
  buffer << chart_json.dump(4) << std::endl;
  const std::string data = buffer.str();
  return data;
}

bool ExecutionRoute::should_redispatch(const std::string& kernel) {
#ifdef HB_REDISPATCH
  if (beacons.find(kernel) != beacons.end()) {
    std::cerr << "at top level kernel " << kernel << std::endl;
    TORCH_INTERNAL_ASSERT(odometer < route.size(), "ERROR: Route is shorter than execution chart");
    auto route_kernel = std::get<0>(route[odometer]);
    auto redispatch = std::get<1>(route[odometer]);
    TORCH_INTERNAL_ASSERT(route_kernel.compare(kernel) == 0,
        "ERROR: Route and execution chart disagree. Expect ", route_kernel, " but found ", kernel);
    odometer++;
    std::cerr << "should I redispatch? 1/0" << std::endl;
    if (redispatch) {
      std::cerr << "redispatching..." << std::endl;
      return true;
    }
  }
#endif
  return false;
}

// ========== ExecutionRoute C10_API ==========

bool route_add_waypoint(const std::string& kernel, bool redispatch) {
#ifdef HB_REDISPATCH
  g_execution_route.add_waypoint(kernel, redispatch);
  return true;
#else
  return false;
#endif
}

bool should_redispatch(const std::string& kernel) {
  return g_execution_route.should_redispatch(kernel);
}

const std::string route_print() {
  return g_execution_route.print();
}

}} // namespace c10::probe
