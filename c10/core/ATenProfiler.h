#pragma once

#include <c10/macros/Macros.h>

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>

namespace c10 {

class ATenProfiler {
public:
  ATenProfiler() = default;
  ~ATenProfiler() { print(); }

  void add_log(const std::vector<std::string>& stack, std::chrono::microseconds time);

private:
  void print();
  std::map<std::vector<std::string>, std::chrono::microseconds> dict;
};

struct C10_API ATenProfilerLog {
public:
  ATenProfilerLog(const std::string& func_name);
  ~ATenProfilerLog();
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

extern ATenProfiler g_aten_profiler;
extern std::vector<std::string> g_curr_call_stack;

#define LogATenKernel() ATenProfilerLog log(__PRETTY_FUNCTION__);

}

