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
std::vector<std::string> traverse;

// ========= AtenProfiler Members ===========

// Mark the beginning of ROI
void ATenProfiler::profiling_start() {
  in_roi = true;
  g_curr_call_stack.clear();
#if defined(PROFILE_ATEN) || defined(PROFILE_UNIMPL)
  std::cerr << " ATen profiler collecting ...";
# ifdef PROFILE_ATEN
  std::cerr << " execution time";
  clear_exeuction_time_dict();
  // mark current time
  time_in_roi = new ExecutionTimeLog();
# endif // ifdef PROFILE_ATEN
# ifdef PROFILE_UNIMPL
  clear_unimpl_kernel();
  std::cerr << " unimplemented kernels";
# endif // ifdef PROFILE_UNIMPL
  std::cerr << std::endl;
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
  std::vector<std::string> fake_roi_stack;
  fake_roi_stack.push_back("time_in_roi");
  time_in_roi->log_self(fake_roi_stack);
  fake_roi_stack.pop_back();
  delete time_in_roi;
#endif
  for (const auto& k : traverse) {
    std::cerr << k << std::endl;
  }
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

// =============== Aten Profiler Log Members =======================

// Entering a function
ATenProfilerLog::ATenProfilerLog(const std::string& func_name)
  : execution_time_log(ExecutionTimeLog())
{
  if (!aten_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back(func_name);
    if (g_curr_call_stack.size() == 1) {
      traverse.push_back(func_name);
    }
  }
}

// Returning from a function
ATenProfilerLog::~ATenProfilerLog()
{
  if (!aten_profiler_in_parallel_region()) {
    execution_time_log.log_self(g_curr_call_stack);
    g_curr_call_stack.pop_back();
  }
}

}} // namespace c10::probe
