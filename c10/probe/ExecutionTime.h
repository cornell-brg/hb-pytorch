#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <time.h>

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
  long diff_microsecond(timespec& start, timespec& end);
private:
  std::map<std::vector<std::string>, std::chrono::microseconds> execution_time_dict;
};

C10_PROBE_API const std::string exec_time_raw_stack();

struct C10_PROBE_API ExecutionTimeLog {
public:
  ExecutionTimeLog(std::vector<std::string>& stack,
                   const std::string& func_name);
  ~ExecutionTimeLog();
private:
  std::vector<std::string>& stack;
  timespec tv;
};

extern ExecutionTimeProfiler g_execution_time_profiler;
extern timespec global_clk;

}} // namespace c10::probe
