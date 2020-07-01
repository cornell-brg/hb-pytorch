#pragma once

#include <c10/probe/ProbeMacros.h>
#include <c10/probe/Unimplemented.h>
#include <c10/probe/ExecutionTime.h>
#include <c10/probe/Chart.h>
#include <c10/probe/Route.h>
#include <c10/probe/Fallback.h>

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>

namespace c10 {
namespace probe {

class HBProfiler {
public:
  HBProfiler() : in_roi(false) {};
  ~HBProfiler() = default;
  void profiling_start();
  void profiling_end();
  bool in_roi;
private:
  ExecutionTimeLog* time_in_roi;
};

C10_PROBE_API void hb_profiler_start();
C10_PROBE_API void hb_profiler_end();
C10_PROBE_API bool hb_profiler_is_in_roi();
C10_PROBE_API bool hb_profiler_is_top_level();
C10_PROBE_API bool hb_profiler_thread_safe();

extern HBProfiler g_hb_profiler;
extern std::vector<std::string> g_curr_call_stack;

struct C10_PROBE_API HBProfilerLog {
public:
  HBProfilerLog(const std::string& func_name);
  ~HBProfilerLog();
private:
  ExecutionTimeLog* execution_time_log;
};

// HBProfilerTrimLog only manipulates g_curr_call_stack
// All data logging needs to be done by hand
struct C10_PROBE_API HBProfilerTrimLog {
public:
  HBProfilerTrimLog();
  ~HBProfilerTrimLog();
  void trim_manual_log_exec_time(std::chrono::microseconds simulated);
};

#define LogATenKernel() HBProfilerLog log(__PRETTY_FUNCTION__);
#define LogATenKernelWithName(aten_profiler_kernel_name) HBProfilerLog log(aten_profiler_kernel_name);

}} // namespace c10::probe
