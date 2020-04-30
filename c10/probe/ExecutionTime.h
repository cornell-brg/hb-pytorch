#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <vector>
#include <string>
#include <chrono>

namespace c10 {
namespace probe {

C10_PROBE_API const std::string aten_profiler_dump();
C10_PROBE_API void aten_profiler_stack_print();
void log_execution_time(const std::vector<std::string>& stack,
                        std::chrono::microseconds time);
void clear_exeuction_time_dict();

struct C10_PROBE_API ExecutionTimeLog {
public:
  ExecutionTimeLog();
  ~ExecutionTimeLog() = default;
  void log_self(const std::vector<std::string>& stack);
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

extern std::map<std::vector<std::string>, std::chrono::microseconds> g_execution_time_dict;

}} // namespace c10::probe
