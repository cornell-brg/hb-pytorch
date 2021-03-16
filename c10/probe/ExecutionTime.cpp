#include <c10/probe/ExecutionTime.h>

#include <iostream>
#include <iomanip>
#include <sstream>

namespace c10 {
namespace probe {

ExecutionTimeProfiler g_execution_time_profiler;
timespec global_clk;

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

const std::string ExecutionTimeProfiler::str_dump() {
  using std::chrono::microseconds;
  using namespace std;
  std::stringstream buffer;

  for (const auto& p : execution_time_dict) {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(p.second);
    auto& stack = p.first;
    // we concat the call stack to get the name
    std::string name;
    for (size_t i = 0; i < stack.size() - 1; i++) {
      name += stack[i];
      name += "<|>";
    }
    name += stack.back();
    buffer << name << ";" << us.count() << endl;
  }
  // buffer.str() is a temp object that will be destroyed
  // at the end of this expression
  const std::string data = buffer.str();
  return data;
}

long ExecutionTimeProfiler::diff_microsecond(timespec& start, timespec& end) {
  long end_microsecond   =   end.tv_sec * 1000000 +   end.tv_nsec / 1000;
  long start_microsecond = start.tv_sec * 1000000 + start.tv_nsec / 1000;
  return (end_microsecond - start_microsecond);
}

// ============ ExecutionTimeProfiler C10_API ============

const std::string exec_time_raw_stack() {
  return g_execution_time_profiler.str_dump();
}

// ============ ExecutionTimeLog Member ============

ExecutionTimeLog::ExecutionTimeLog(std::vector<std::string>& stack,
                                   const std::string& func_name) : stack(stack) {
  // stop the previous clock
  clock_gettime(CLOCK_MONOTONIC, &tv);
  // time belongs to the upper level
  std::chrono::microseconds delta(g_execution_time_profiler.diff_microsecond(global_clk, tv));
  g_execution_time_profiler.log(stack, delta);
  // extend stack with funcName
  stack.push_back(func_name);
  // start the clock
  clock_gettime(CLOCK_MONOTONIC, &global_clk);
}

ExecutionTimeLog::~ExecutionTimeLog() {
  // stop the clock
  clock_gettime(CLOCK_MONOTONIC, &tv);
  // time belongs to current level
  std::chrono::microseconds delta(g_execution_time_profiler.diff_microsecond(global_clk, tv));
  g_execution_time_profiler.log(stack, delta);
  // remote self from stack
  stack.pop_back();
  // start the clock
  clock_gettime(CLOCK_MONOTONIC, &global_clk);
}

}} // namespace c10::probe
