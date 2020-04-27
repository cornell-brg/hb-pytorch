#include <kernel_common.hpp>
#include <kernel_log.h>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

KernelLogger::KernelLogger(bool on, std::string log_path) :
  on(on),
  log_path(log_path) {
    log_json = json();
  }

void KernelLogger::add_kernel(const char* kernel) {
  std::cout << "Logging " << kernel << std::endl;
}
