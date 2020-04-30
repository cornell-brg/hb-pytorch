#include <c10/probe/Unimplemented.h>
#include <sstream>

namespace c10 {
namespace probe {

std::map<std::string, long> unimpl_kernel;

void log_unimpl_kernel(const std::string& kernel) {
  if (unimpl_kernel.find(kernel) != unimpl_kernel.end()) {
    unimpl_kernel[kernel] += 1;
  } else {
    unimpl_kernel[kernel] = 1;
  }
}

void aten_profiler_unimpl_print() {
  using namespace std;

  std::cerr << "==========================================================================" << std::endl;
  std::cerr << " Native kernels that are used but not implemented for HammerBlade:" << std::endl;
  std::cerr << "==========================================================================" << std::endl;
  cerr << setw(180) << std::left << "Function" << "   " << "Times" << endl;
  for (const auto& k : unimpl_kernel) {
    cerr << setw(180) << std::left << k.first << "   " << k.second << endl;
  }
}

void clear_unimpl_kernel() {
  unimpl_kernel.clear();
}
