#include <kernel_common.hpp>
#include <iostream>

void kernel_log(const char* kernel) {
  std::cout << "Logging " << kernel << std::endl;
}

void kernel_log(hb_tensor_t* arg) {
  std::cout << "  tensor" << std::endl;
}

void kernel_log(hb_vector_t* arg) {
  std::cout << "  vector" << std::endl;
}

void kernel_log(float* arg) {
  std::cout << "  float" << std::endl;
}

void kernel_log(int32_t* arg) {
  std::cout << "  int32_t" << std::endl;
}

void kernel_log(uint32_t* arg) {
  std::cout << "  uint32_t" << std::endl;
}
