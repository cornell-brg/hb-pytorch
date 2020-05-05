#pragma once

#include <c10/probe/ProbeMacros.h>
#include <c10/probe/Unimplemented.h>
#include <c10/probe/ExecutionTime.h>
#include <c10/probe/Chart.h>
#include <c10/probe/Route.h>

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>

namespace c10 {
namespace probe {

class ATenProfiler {
public:
  ATenProfiler() : in_roi(false) {};
  ~ATenProfiler() = default;
  void profiling_start();
  void profiling_end();
  bool in_roi;
private:
  ExecutionTimeLog* time_in_roi;
};

C10_PROBE_API void aten_profiler_start();
C10_PROBE_API void aten_profiler_end();
C10_PROBE_API bool is_in_aten_profiler_roi();
C10_PROBE_API bool is_top_level_kernel();

struct C10_PROBE_API ATenProfilerLog {
public:
  ATenProfilerLog(const std::string& func_name);
  ~ATenProfilerLog();
private:
  ExecutionTimeLog* execution_time_log;
};

extern ATenProfiler g_aten_profiler;
extern std::vector<std::string> g_curr_call_stack;

#define LogATenKernel() ATenProfilerLog log(__PRETTY_FUNCTION__);
#define LogATenKernelWithName(aten_profiler_kernel_name) ATenProfilerLog log(aten_profiler_kernel_name);

}} // namespace c10::probe
