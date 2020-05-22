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
  using namespace std;
  std::stringstream buffer;

  for (const auto& p : execution_time_dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;
    // we concat the call stack to get the name
    std::string name;
    for (size_t i = 0; i < stack.size() - 1; i++) {
      name += stack[i];
      name += "<|>";
    }
    name += stack.back();
    buffer << name << ";" << ms << endl;
  }
  // buffer.str() is a temp object that will be destroyed
  // at the end of this expression
  const std::string data = buffer.str();
  return data;
}

// ============ ExecutionTimeProfiler C10_API ============

const std::string exec_time_raw_stack() {
  return g_execution_time_profiler.str_dump();
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
