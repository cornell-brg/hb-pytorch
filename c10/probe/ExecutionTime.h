#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <vector>
#include <string>
#include <chrono>

namespace c10 {
namespace probe {

class ExecutionTimeProfiler {
public:
  ExecutionTimeProfiler() = default;
  ~ExecutionTimeProfiler() = default;
  void reset();
  void log(const std::vector<std::string>& stack,
           std::chrono::microseconds time);
  const std::string str_dump();
private:
  std::map<std::vector<std::string>, std::chrono::microseconds> execution_time_dict;
};

C10_PROBE_API const std::string exec_time_raw_stack();

struct C10_PROBE_API ExecutionTimeLog {
public:
  ExecutionTimeLog(const std::vector<std::string>& stack);
  ~ExecutionTimeLog();
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  const std::vector<std::string> stack;
};

extern ExecutionTimeProfiler g_execution_time_profiler;

}} // namespace c10::probe
