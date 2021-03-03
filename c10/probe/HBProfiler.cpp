#include <c10/probe/HBProfiler.h>

#include <map>
#include <vector>
#include <sstream>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace c10 {
namespace probe {

// Global Variables
HBProfiler g_hb_profiler;
std::vector<std::string> g_curr_call_stack;

// ========= AtenProfiler Members ===========

// Mark the beginning of ROI
void HBProfiler::profiling_start() {
  in_roi = true;
  g_curr_call_stack.clear();
#ifdef PROFILE_ATEN
  std::cerr << " ATen profiler collecting ..." << std::endl;
  // reset profiler pluggins
  g_unimpl_kernel_profiler.reset();
  g_execution_time_profiler.reset();
  g_execution_charter.reset();
  // mark current time
  g_curr_call_stack.push_back("ROI");
  // start the clock
  clock_gettime(CLOCK_MONOTONIC, &global_clk);
#else
  std::cerr << "Warning: ATen profiler is invoked "
            << "but PyTorch is not built with profiling capability "
            << "ROI entry is still marked"
            << std::endl;
#endif
  return;
}

// Mark the end of ROI
void HBProfiler::profiling_end() {
  in_roi = false;
#ifdef PROFILE_ATEN
  // stop the clock
  timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  std::chrono::microseconds delta(g_execution_time_profiler.diff_microsecond(global_clk, tv));
  g_execution_time_profiler.log(g_curr_call_stack, delta);
  g_curr_call_stack.pop_back();
  // we should have just popped ROI
  assert(g_curr_call_stack.size() == 0);
#endif
#ifdef HB_REDISPATCH
  g_execution_charter.print();
#endif
  return;
}


// =============== c10 probe internal helper functions ========================

bool hb_profiler_thread_safe() {
#ifdef _OPENMP
  // we profile if the current function is not in an active parallel region
  // OR thread id == 0
  if (!omp_in_parallel() || (omp_get_thread_num() == 0)) {
    return true;
  } else {
    return false;
  }
#else
  // if not compiled with OMP, we have no idea if the current thread should be
  // profiled. use at your own risk
  return true;
#endif
}

// =============== c10 probe API functions ========================

void hb_profiler_start() {
  g_hb_profiler.profiling_start();
  return;
}

void hb_profiler_end() {
  g_hb_profiler.profiling_end();
  return;
}

bool hb_profiler_is_in_roi() {
  return g_hb_profiler.in_roi;
}

bool hb_profiler_is_top_level() {
  return (g_curr_call_stack.size() == 2);
}

// =============== HBProfilerLog Members =======================

// Entering a function
HBProfilerLog::HBProfilerLog(const std::string& func_name) {
  if (hb_profiler_is_in_roi() && hb_profiler_thread_safe()) {
    execution_time_log = new ExecutionTimeLog(g_curr_call_stack, func_name);
#ifdef HB_REDISPATCH
    if (hb_profiler_is_top_level()) {
      g_execution_charter.log(func_name);
    }
#endif
  }
}

// Returning from a function
HBProfilerLog::~HBProfilerLog()
{
  if (hb_profiler_is_in_roi() && hb_profiler_thread_safe()) {
    delete execution_time_log;
  }
}

// =============== HBProfilerTrimLog Members =======================

// Entering a function
HBProfilerTrimLog::HBProfilerTrimLog() {
  if (hb_profiler_is_in_roi() && hb_profiler_thread_safe()) {
    g_curr_call_stack.push_back("@TRIM@");
  }
}

// Returning from a function
HBProfilerTrimLog::~HBProfilerTrimLog()
{
  if (hb_profiler_is_in_roi() && hb_profiler_thread_safe()) {
    g_curr_call_stack.pop_back();
  }
}

void HBProfilerTrimLog::trim_manual_log_exec_time(std::chrono::microseconds simulated) {
  if (hb_profiler_is_in_roi() && hb_profiler_thread_safe()) {
    g_execution_time_profiler.log(g_curr_call_stack, simulated);
  }
}

}} // namespace c10::probe
