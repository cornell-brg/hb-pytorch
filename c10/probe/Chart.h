#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

C10_PROBE_API void aten_profiler_execution_chart_print();
void log_execution_chart(const std::string& kernel);
void clear_execution_chart();
void clear_kernels_of_interest();
void add_kernels_of_interest(const std::string& kernel);

//hack
C10_PROBE_API bool should_redispatch(const std::string& kernel);

extern std::vector<std::string> execution_chart;
extern std::map<std::string, bool> kernels_of_interest;

}} // namespace c10::probe
