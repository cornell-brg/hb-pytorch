#include <c10/probe/Chart.h>

#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace c10 {
namespace probe {

ExecutionCharter g_execution_charter;

// ========== ExecutionCharter Member ==========

void ExecutionCharter::reset() {
  execution_chart.clear();
  beacons.clear();
  // hack
  add_beacon("at::Tensor at::TypeDefault::embedding(const at::Tensor&, const at::Tensor&, int64_t, bool, bool)");
  add_beacon("at::Tensor at::TypeDefault::sum(const at::Tensor&, c10::IntArrayRef, bool, c10::optional<c10::ScalarType>)");
  add_beacon("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)");
  add_beacon("at::Tensor at::CPUType::{anonymous}::mm(const at::Tensor&, const at::Tensor&)");
  add_beacon("at::Tensor at::TypeDefault::embedding_backward(const at::Tensor&, const at::Tensor&, int64_t, int64_t, bool, bool)");
}

void ExecutionCharter::log(const std::string& kernel) {
  if (beacons.find(kernel) != beacons.end()) {
    execution_chart.push_back(kernel);
  }
}

void ExecutionCharter::add_beacon(const std::string& kernel) {
  beacons[kernel] = true;
}

void ExecutionCharter::print() {
  json chart_json = json::array();
  for (const auto& k : execution_chart) {
    //std::cerr << k << std::endl;
    //std::cerr << " || " << std::endl;
    json kernel_json;
    kernel_json["signature"] = k;
    kernel_json["offload"] = false;
    chart_json.push_back(kernel_json);
  }
  std::cerr << chart_json.dump(4) << std::endl;
}

bool ExecutionCharter::should_redispatch(const std::string& kernel) {
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

// ========== ExecutionCharter C10_API ==========

void aten_profiler_execution_chart_print() {
  g_execution_charter.print();
}

// hack
bool should_redispatch(const std::string& kernel) {
  return g_execution_charter.should_redispatch(kernel);
}

}} // namespace c10::probe
