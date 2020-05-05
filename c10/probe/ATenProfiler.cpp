#include <c10/probe/ATenProfiler.h>

#include <map>
#include <vector>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace c10 {
namespace probe {

// Global Variables
ATenProfiler g_aten_profiler;
std::vector<std::string> g_curr_call_stack;

// ========= AtenProfiler Members ===========

// Mark the beginning of ROI
void ATenProfiler::profiling_start() {
  in_roi = true;
  g_curr_call_stack.clear();
#ifdef PROFILE_ATEN
  std::cerr << " ATen profiler collecting ..." << std::endl;
  // reset profiler pluggins
  g_unimpl_kernel_profiler.reset();
  g_execution_time_profiler.reset();
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
void ATenProfiler::profiling_end() {
  in_roi = false;
#ifdef PROFILE_ATEN
  delete time_in_roi;
  g_execution_charter.print();
#endif
  return;
}


// =============== c10 probe API functions ========================

bool aten_profiler_in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

void aten_profiler_start() {
  g_aten_profiler.profiling_start();
  return;
}

void aten_profiler_end() {
  g_aten_profiler.profiling_end();
  return;
}

bool is_in_aten_profiler_roi() {
  return g_aten_profiler.in_roi;
}

bool is_top_level_kernel() {
  return (g_curr_call_stack.size() == 1);
}

// =============== Aten Profiler Log Members =======================

// Entering a function
ATenProfilerLog::ATenProfilerLog(const std::string& func_name) {
  if (!aten_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back(func_name);
    execution_time_log = new ExecutionTimeLog(g_curr_call_stack);
    if (is_top_level_kernel()) {
      g_execution_charter.log(func_name);
    }
  }
}

// Returning from a function
ATenProfilerLog::~ATenProfilerLog()
{
  if (!aten_profiler_in_parallel_region()) {
    delete execution_time_log;
    g_curr_call_stack.pop_back();
  }
}

}} // namespace c10::probe
