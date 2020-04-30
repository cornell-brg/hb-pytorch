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

// Print low level call stack break down
void ATenProfiler::print()
{
  using std::chrono::microseconds;
  using namespace std;

  cerr << setw(180) << std::left << "Function" << "   " << "Time" << endl;
  double total_time = 0.0;
  for (const auto& p : dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;
    // we concat the call stack to get the name
    std::string name;

    // if (stack.size() > 1)
    //   continue; // do not print sub-routine for now...
    if (stack.size() == 1) {
      total_time += ms;
    }

    if (stack.size() > 1) {
      for (size_t i = 1; i < stack.size(); i++) {
        name += "  ";
      }
      name += "|- ";
    }
    name += stack.back();
    cerr << setw(180) << std::left << name << "   " << ms / 1000.0 << " s" << endl;
  }
  cerr << setw(180) << std::left << "Aggregated total:" << "   " << total_time / 1000.0 << " s" << endl;
}

// Add a log entry
void ATenProfiler::add_log(const std::vector<std::string>& stack, std::chrono::microseconds time) {
  if (dict.find(stack) != dict.end()) {
    dict[stack] += time;
  } else {
    dict[stack] = time;
  }
}

// Add a unimplemented log entry
void ATenProfiler::add_kernel_log(const std::string& kernel) {
  if (unimpl_kernel.find(kernel) != unimpl_kernel.end()) {
    unimpl_kernel[kernel] += 1;
  } else {
    unimpl_kernel[kernel] = 1;
  }
}

// Mark the beginning of ROI
void ATenProfiler::profiling_start() {
  in_roi = true;
#if defined(PROFILE_ATEN) || defined(PROFILE_UNIMPL)
  std::cerr << " ATen profiler collecting ...";
# ifdef PROFILE_ATEN
  std::cerr << " execution time";
  // clear the dict when entering ROI
  g_curr_call_stack.clear();
  dict.clear();
  // mark current time
  start = std::chrono::high_resolution_clock::now();
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
  auto delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
  // total time in ROI, measured in ms.
  time_in_roi = delta.count() / 1000.0;
#endif
  return;
}

// Convert profiling log to string so we can move it to
// Python world
const std::string ATenProfiler::profiling_dump() {

  using std::chrono::microseconds;

  std::stringstream buffer;
  double total_time = 0.0;

  // raw string dump starts with time in RIO
  buffer << "time_in_roi" << ";" << time_in_roi << std::endl;

  for (const auto& p : dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;

    if (stack.size() == 1) {
      total_time += ms;
    } else {
      continue;
    }

    buffer << stack.back() << ";" << ms << std::endl;
  }

  // raw string dump ends with aggregated total
  buffer << "agg_total" << ";" << total_time << std::endl;

  // buffer.str() is a temp object that will be destroyed
  // at the end of this expression
  const std::string data = buffer.str();
  return data;
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

const std::string aten_profiler_dump() {
  return g_aten_profiler.profiling_dump();
}

bool is_in_aten_profiler_roi() {
  return g_aten_profiler.in_roi;
}

void aten_profiler_stack_print() {
  g_aten_profiler.print();
  return;
}

// =============== Aten Profiler Log Members =======================

// Entering a function
ATenProfilerLog::ATenProfilerLog(const std::string& func_name)
  : start(std::chrono::high_resolution_clock::now())
{
  if (!aten_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back(func_name);
  }
}

// Returning from a function
ATenProfilerLog::~ATenProfilerLog()
{
  if (!aten_profiler_in_parallel_region()) {
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    g_aten_profiler.add_log(g_curr_call_stack, delta);
    g_curr_call_stack.pop_back();
  }
}

}} // namespace c10::probe
