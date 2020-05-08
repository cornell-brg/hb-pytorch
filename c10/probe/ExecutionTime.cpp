#include <c10/probe/ExecutionTime.h>

#include <iostream>
#include <iomanip>
#include <sstream>

namespace c10 {
namespace probe {

ExecutionTimeProfiler g_execution_time_profiler;

// ============ ExecutionTimeProfiler Members ============

void ExecutionTimeProfiler::reset() {
  execution_time_dict.clear();
}

void ExecutionTimeProfiler::log(const std::vector<std::string>& stack,
         std::chrono::microseconds time) {
  if (execution_time_dict.find(stack) != execution_time_dict.end()) {
    execution_time_dict[stack] += time;
  } else {
    execution_time_dict[stack] = time;
  }
}

// ============ ExecutionTimeLog Member ============

const std::string ExecutionTimeProfiler::str_dump() {
  using std::chrono::microseconds;
  std::stringstream buffer;
  double total_time = 0.0;
  // raw string dump starts with time in RIO
  std::vector<std::string> fake_roi_stack;
  fake_roi_stack.push_back("time_in_roi");
  buffer << "time_in_roi" << ";"
         << execution_time_dict[fake_roi_stack].count() / 1000.0 << std::endl;
  fake_roi_stack.pop_back();

  for (const auto& p : execution_time_dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;
    if (stack.size() == 1 && stack.back().compare("time_in_roi") != 0) {
      total_time += ms;
    } else {
      continue;
    }
    buffer << stack.back() << ";" << ms << std::endl;
  }

  // raw string dump ends with aggregated total
  buffer << "agg_total" << ";" << total_time << std::endl;
  // buffer.str() is a temp object that will be destroyed
  // at the end of this expression
  const std::string data = buffer.str();
  return data;
}

const std::string ExecutionTimeProfiler::stack_print() {
  using std::chrono::microseconds;
  std::stringstream buffer;
  using namespace std;
  buffer << setw(180) << std::left << "Function" << "   " << "Time" << endl;
  double total_time = 0.0;

  for (const auto& p : execution_time_dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;
    // we concat the call stack to get the name
    std::string name;
    if (stack.size() == 1 && stack.back().compare("time_in_roi") != 0) {
      total_time += ms;
    }
    if (stack.size() > 1) {
      for (size_t i = 1; i < stack.size(); i++) {
        name += "  ";
      }
      name += "|- ";
    }
    name += stack.back();
    buffer << setw(180) << std::left << name << "   " << ms / 1000.0 << " s" << endl;
  }

  buffer << setw(180) << std::left << "Aggregated total:" << "   " << total_time / 1000.0 << " s" << endl;
  const std::string data = buffer.str();
  return data;
}

// ============ ExecutionTimeProfiler C10_API ============

const std::string exec_time_fancy_table() {
  return g_execution_time_profiler.str_dump();
}

const std::string exec_time_raw_stack() {
  return g_execution_time_profiler.stack_print();
}

// ============ ExecutionTimeLog Member ============

ExecutionTimeLog::ExecutionTimeLog(const std::vector<std::string>& stack)
  : start(std::chrono::high_resolution_clock::now()),
    stack(stack) {}

ExecutionTimeLog::~ExecutionTimeLog() {
  auto delta = std::chrono::duration_cast<std::chrono::microseconds>
    (std::chrono::high_resolution_clock::now() - start);
  g_execution_time_profiler.log(stack, delta);
}

}} // namespace c10::probe
