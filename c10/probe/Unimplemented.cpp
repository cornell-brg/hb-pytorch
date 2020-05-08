#include <c10/probe/Unimplemented.h>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace c10 {
namespace probe {

UnimplKernelProfiler g_unimpl_kernel_profiler;

void UnimplKernelProfiler::reset() {
  unimpl_kernel.clear();
}

void UnimplKernelProfiler::log(const std::string& kernel) {
  if (unimpl_kernel.find(kernel) != unimpl_kernel.end()) {
    unimpl_kernel[kernel] += 1;
  } else {
    unimpl_kernel[kernel] = 1;
  }
}

const std::string UnimplKernelProfiler::print() {
  using namespace std;
  stringstream buffer;
  buffer << "==========================================================================" << endl;
  buffer << " Native kernels that are used but not implemented for HammerBlade:" << endl;
  buffer << "==========================================================================" << endl;
  buffer << setw(180) << left << "Function" << "   " << "Times" << endl;
  for (const auto& k : unimpl_kernel) {
    buffer << setw(180) << left << k.first << "   " << k.second << endl;
  }
  const std::string data = buffer.str();
  return data;
}

const std::string unimpl_raw_print() {
  return g_unimpl_kernel_profiler.print();
}

void log_unimpl_kernel(const std::string& kernel) {
  g_unimpl_kernel_profiler.log(kernel);
}

}} // namespace c10::probe
