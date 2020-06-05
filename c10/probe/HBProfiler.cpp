#include <c10/probe/HBProfiler.h>

#include <map>
#include <vector>
#include <sstream>
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
  g_per_op_execution_time_profiler.reset();
  g_execution_charter.reset();
  // mark current time
  std::vector<std::string> fake_roi_stack;
  fake_roi_stack.push_back("time_in_roi");
  time_in_roi = new ExecutionTimeLog(fake_roi_stack);
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
  delete time_in_roi;
  g_execution_charter.print();
#endif
  return;
}


// =============== c10 probe API functions ========================

bool hb_profiler_in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

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
  return (g_curr_call_stack.size() == 1);
}

// =============== HBProfilerLog Members =======================

// Entering a function
HBProfilerLog::HBProfilerLog(const std::string& func_name) {
  if (hb_profiler_is_in_roi() && !hb_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back(func_name);
    execution_time_log = new ExecutionTimeLog(g_curr_call_stack);
    if (hb_profiler_is_top_level()) {
      g_per_op_execution_time_profiler.reset();
      g_execution_charter.log(func_name);
    }
  }
}

// Returning from a function
HBProfilerLog::~HBProfilerLog()
{
  if (hb_profiler_is_in_roi() && !hb_profiler_in_parallel_region()) {
    delete execution_time_log;
    if (hb_profiler_is_top_level()) {
      std::string per_op_log = g_per_op_execution_time_profiler.str_dump();
      if (per_op_log.find("@CPU_LOG@") != std::string::npos) {
        std::cerr << "#TOP_LEVEL_FUNC#__" << g_curr_call_stack.back() << std::endl;
        std::cerr << per_op_log << std::endl;
        std::cerr << "#TOP_LEVEL_FUNC_END#__" << g_curr_call_stack.back() << std::endl;
      }
    }
    g_curr_call_stack.pop_back();
  }
}

// =============== HBProfilerTrimLog Members =======================

// Entering a function
HBProfilerTrimLog::HBProfilerTrimLog() {
  if (hb_profiler_is_in_roi() && !hb_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back("@TRIM@");
  }
}

// Returning from a function
HBProfilerTrimLog::~HBProfilerTrimLog()
{
  if (hb_profiler_is_in_roi() && !hb_profiler_in_parallel_region()) {
    g_curr_call_stack.pop_back();
  }
}

void HBProfilerTrimLog::trim_manual_log_exec_time(std::chrono::microseconds simulated) {
  g_execution_time_profiler.log(g_curr_call_stack, simulated);
  g_per_op_execution_time_profiler.log(g_curr_call_stack, simulated);
}

}} // namespace c10::probe
