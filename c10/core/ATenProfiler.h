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
  ATenProfiler() : in_roi(false) {};
  ~ATenProfiler() = default;
  void add_log(const std::vector<std::string>& stack, std::chrono::microseconds time);
  void add_kernel_log(const std::string& kernel);
  void profiling_start();
  void profiling_end();
  bool in_roi;

private:
  std::map<std::vector<std::string>, std::chrono::microseconds> dict;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::map<std::string, long> unimpl_kernel;
  void print();
  void print_unimpl_kernel();
};

C10_API void aten_profiler_start();
C10_API void aten_profiler_end();
C10_API bool is_in_aten_profiler_roi();
C10_API void log_unimpl_kernel(const std::string& kernel);

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

