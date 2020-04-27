#include <kernel_common.hpp>
#include <kernel_log.h>
#include <iostream>

// A popular C++ library for json pasrsing and
// serialization. Inlcuded as a header only library.
//
// Source: https://github.com/nlohmann/json
// Release: https://github.com/nlohmann/json/releases/tag/v3.7.3
#include <json.hpp>

using json = nlohmann::json;

void KernelLogger::add_kernel(const char* kernel) {
  std::cout << "Logging " << kernel << std::endl;
}
