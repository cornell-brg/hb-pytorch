#pragma once

#include <c10/macros/Macros.h>

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>

namespace c10 {

class ATenProfiler {
public:
  ATenProfiler() = default;
  ~ATenProfiler() = default;
  void add_log(const std::vector<std::string>& stack, std::chrono::microseconds time);
  void profiling_start();
  void profiling_end();

private:
  std::map<std::vector<std::string>, std::chrono::microseconds> dict;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  void print();
};

C10_API void aten_profiler_start();
C10_API void aten_profiler_end();

struct C10_API ATenProfilerLog {
public:
  ATenProfilerLog(const std::string& func_name);
  ~ATenProfilerLog();
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

extern ATenProfiler g_aten_profiler;
extern std::vector<std::string> g_curr_call_stack;

#define LogATenKernel() ATenProfilerLog log(__PRETTY_FUNCTION__);

}

