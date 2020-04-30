#include <c10/probe/Chart.h>

#include <iostream>

namespace c10 {
namespace probe {

std::vector<std::string> execution_chart;
std::map<std::string, bool> kernels_of_interest;

void aten_profiler_execution_chart_print() {
  for (const auto& k : execution_chart) {
    std::cerr << k << std::endl;
    std::cerr << " || " << std::endl;
  }
}

void log_execution_chart(const std::vector<std::string>& stack) {
  // log top level kernels only
  if (stack.size() == 1) {
    auto kernel = stack.back();
    if (kernels_of_interest.find(kernel) != kernels_of_interest.end()) {
      execution_chart.push_back(kernel);
    }
  }
}

void clear_execution_chart() {
  execution_chart.clear();
}

void clear_kernels_of_interest() {
  kernels_of_interest.clear();
}

void add_kernels_of_interest(const std::string kernel) {
  kernels_of_interest[kernel] = true;
}

}} // namespace c10::probe
