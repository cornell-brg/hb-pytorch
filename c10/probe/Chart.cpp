#include <c10/probe/Chart.h>

#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace c10 {
namespace probe {

std::vector<std::string> execution_chart;
std::map<std::string, bool> kernels_of_interest;

void aten_profiler_execution_chart_print() {
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

void log_execution_chart(const std::string& kernel) {
  if (kernels_of_interest.find(kernel) != kernels_of_interest.end()) {
    execution_chart.push_back(kernel);
  }
}

void clear_execution_chart() {
  execution_chart.clear();
}

void clear_kernels_of_interest() {
  kernels_of_interest.clear();
}

void add_kernels_of_interest(const std::string& kernel) {
  kernels_of_interest[kernel] = true;
}

bool should_redispatch(const std::string& kernel) {
  if (kernels_of_interest.find(kernel) != kernels_of_interest.end()) {
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

}} // namespace c10::probe
