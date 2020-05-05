#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

class UnimplKernelProfiler {
public:
  UnimplKernelProfiler() = default;
  ~UnimplKernelProfiler() = default;
  void reset();
  void log(const std::string& kernel);
  void print();
private:
  std::map<std::string, long> unimpl_kernel;
};

C10_PROBE_API void log_unimpl_kernel(const std::string& kernel);
C10_PROBE_API void aten_profiler_unimpl_print();

extern UnimplKernelProfiler g_unimpl_kernel_profiler;

}} // namespace c10::probe
