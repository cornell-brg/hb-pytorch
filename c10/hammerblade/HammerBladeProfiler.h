//=================================================================
// HB Profiler
// 05/03/2020 Bandhav Veluri
//=================================================================

#include <map>
#include <string>
#include <array>
#include <bsg_manycore_printing.h>

namespace c10 {
namespace hammerblade {

class HBProfiler {
  // Cummulative time consumed for executing all kernels
  uint32_t execution_time;

  // <kernel_name>: {start_time, end_time, cummulative}
  std::map<std::string, std::array<uint64_t, 3>> profile_log;

  public:
    HBProfiler() : 
      execution_time(0) {}

    void kernel_start(const char* kernel) {
      if(profile_log.find(kernel) == profile_log.end()) {
        profile_log[kernel] = {0, 0, 0};
      }

      profile_log[kernel][0] = bsg_time();
    }

    void kernel_end(const char* kernel) {
      TORCH_CHECK(
        profile_log.find(kernel) != profile_log.end(),
        "Can't find the kernel in the log. Please call kernel_start first");

      profile_log[kernel][1] = bsg_time();

      uint64_t kernel_exec_time = profile_log[kernel][1] - 
                            profile_log[kernel][0];
      profile_log[kernel][2] += kernel_exec_time;
      execution_time += kernel_exec_time;
    }
};

}} // namepsace c10::hammerblade
