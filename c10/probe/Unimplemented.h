#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

C10_PROBE_API void log_unimpl_kernel(const std::string& kernel);
C10_PROBE_API void aten_profiler_unimpl_print();
void clear_unimpl_kernel();

extern std::map<std::string, long> unimpl_kernel;

}} // namespace c10::probe
