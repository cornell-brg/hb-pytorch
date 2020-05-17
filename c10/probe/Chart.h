#pragma once

#include <c10/probe/ProbeMacros.h>

#include <map>
#include <string>
#include <vector>

namespace c10 {
namespace probe {

class ExecutionCharter {
public:
  ExecutionCharter() = default;
  ~ExecutionCharter() = default;
  void reset();
  void log(const std::string& kernel);
  void add_beacon(const std::string& kernel);
  void clear_beacon();
  const std::string print();
private:
  std::vector<std::string> execution_chart;
  std::map<std::string, bool> beacons;
};

C10_PROBE_API void chart_add_beacon(const std::string& kernel);
C10_PROBE_API void chart_clear_beacon();
C10_PROBE_API const std::string chart_print();

extern ExecutionCharter g_execution_charter;

}} // namespace c10::probe
