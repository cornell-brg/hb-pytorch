//=================================================================
// HB Profiler
// 05/03/2020 Bandhav Veluri
//=================================================================

#include <string>
#include <c10/hammerblade/HammerBladeException.h>
#include <bsg_manycore_printing.h>
#include "HammerBladeProfiler.h"

namespace c10 {
namespace hammerblade {

void HBProfiler::kernel_start(const char* kernel) {
  if(profile_log.find(kernel) == profile_log.end()) {
    profile_log[kernel] = {0};
  }

  profile_log[kernel][START_TIME] = bsg_time();
}

void HBProfiler::kernel_end(const char* kernel) {
  TORCH_INTERNAL_ASSERT(
    profile_log.find(kernel) != profile_log.end(),
    "Can't find the kernel in profiler_log. ",
    "Please call kernel_start first");

  profile_log[kernel][NUM_CALLS] += 1;
  profile_log[kernel][END_TIME] = bsg_time();

  uint64_t kernel_exec_time = profile_log[kernel][1] - 
                                profile_log[kernel][0];
  profile_log[kernel][CUMMULATIVE] += kernel_exec_time;
  execution_time += kernel_exec_time;
}

std::string HBProfiler::summary() {}

}} // namespace c10::hammerblade
