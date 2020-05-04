//=================================================================
// HB Profiler
// 05/03/2020 Bandhav Veluri
//=================================================================

#pragma once

#include <map>
#include <array>

namespace c10 {
namespace hammerblade {

enum KernelProfileItem {
  NUM_CALLS,    // number of calls to this kernel
  CUMMULATIVE,  // cummulative time for executing this kernel
  START_TIME,   // start time of current kernel call
  END_TIME,     // end time if current kernel call
  NUM_ITEMS
};

/**
 * HammerBlade Execution Profiler
 *
 * Uses cudalite runtime's `bsg_time` api function to aggregate
 * number of cycles each kernel takes. Can be extended to support
 * more precise profiling, when new profiling api functions are
 * available in cudalite runtime.
 */
class HBProfiler {
  // Runtime switch to toggle profiling
  bool on;

  // Cummulative time consumed for executing all kernels
  uint32_t execution_time;

  // <kernel_name>: {<kernel profile items>}
  std::map<std::string, std::array<uint64_t, NUM_ITEMS>> profile_log;

  public:
    HBProfiler() : 
      on(false),
      execution_time(0) {}

    /**
     * Enable profiler
     */
    void enable() {
      on = true;
    }

    /**
     * Disable profiler
     */
    void disable() {
      on = false;
    }

    /**
     * Clear profiler
     */
    void clear() {
      profile_log.clear();
    }

    /**
     * Registers start time of a kernel call.
     */
    void kernel_start(const char* kernel);

    /**
     * Registers end time of a kernel call. Must follow
     * a call to `kernel_start`.
     */
    void kernel_end(const char* kernel);

    /**
     * Returns a formatted string with summary of HB execution
     * until this point.
     */
    std::string summary();
};

}} // namepsace c10::hammerblade
