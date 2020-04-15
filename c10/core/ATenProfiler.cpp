#include <c10/core/ATenProfiler.h>

#include <map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace c10 {

ATenProfiler g_aten_profiler;
std::vector<std::string> g_curr_call_stack;

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

void ATenProfiler::print_unimpl_kernel() {
  using namespace std;

  std::cerr << "==========================================================================" << std::endl;
  std::cerr << " Native kernels that are used but not implemented for HammerBlade:" << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  cerr << setw(180) << std::left << "Function" << "   " << "Times" << endl;
  for (const auto& k : unimpl_kernel) {
    cerr << setw(180) << std::left << k.first << "   " << k.second << endl;
  }
}

void ATenProfiler::add_log(const std::vector<std::string>& stack, std::chrono::microseconds time) {
  if (dict.find(stack) != dict.end()) {
    dict[stack] += time;
  } else {
    dict[stack] = time;
  }
}

void ATenProfiler::add_kernel_log(const std::string& kernel) {
  if (unimpl_kernel.find(kernel) != unimpl_kernel.end()) {
    unimpl_kernel[kernel] += 1;
  } else {
    unimpl_kernel[kernel] = 1;
  }
}

void ATenProfiler::profiling_start() {
#ifdef PROFILE_ATEN
  std::cerr << "==========================================================================" << std::endl;
  std::cerr << " ATen profiler collecting ..." << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  // clear the dict when entering ROI
  g_curr_call_stack.clear();
  dict.clear();
  unimpl_kernel.clear();
  // mark current time
  start = std::chrono::high_resolution_clock::now();
  in_roi = true;
#else
  std::cerr << "Warning: ATen profiler is invoked "
            << "but PyTorch is not built with profiling capability"
            << std::endl;
#endif
  return;
}

void ATenProfiler::profiling_end() {
#ifdef PROFILE_ATEN
  in_roi = false;
  auto delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
  std::cerr << std::endl << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  std::cerr << " ATen profile results" << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  std::cerr << std::setw(180) << std::left << " Total time in ROI:" << "   "
            << delta.count() / 1000000.0 << " s" << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  print();
#endif
#ifdef PROFILE_UNIMPL
  print_unimpl_kernel();
#endif
  return;
}

// ============================================================================

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

void log_unimpl_kernel(const std::string& kernel) {
  g_aten_profiler.add_kernel_log(kernel);
}


ATenProfilerLog::ATenProfilerLog(const std::string& func_name)
  : start(std::chrono::high_resolution_clock::now())
{
  if (!aten_profiler_in_parallel_region()) {
    g_curr_call_stack.push_back(func_name);
  }
}

ATenProfilerLog::~ATenProfilerLog()
{
  if (!aten_profiler_in_parallel_region()) {
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    g_aten_profiler.add_log(g_curr_call_stack, delta);
    g_curr_call_stack.pop_back();
  }
}

}
