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
}

void ExecutionCharter::log(const std::string& kernel) {
  if (beacons.find(kernel) != beacons.end()) {
    execution_chart.push_back(kernel);
  }
}

void ExecutionCharter::add_beacon(const std::string& kernel) {
  beacons[kernel] = true;
}

void ExecutionCharter::clear_beacon() {
  beacons.clear();
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

// ========== ExecutionCharter C10_API ==========

void aten_profiler_execution_chart_print() {
  g_execution_charter.print();
}

void chart_add_beacon(const std::string& kernel) {
  g_execution_charter.add_beacon(kernel);
}

void chart_clear_beacon() {
  g_execution_charter.clear_beacon();
}

}} // namespace c10::probe
