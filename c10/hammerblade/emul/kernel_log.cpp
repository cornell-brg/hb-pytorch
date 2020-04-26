#include <kernel_common.hpp>
#include <iostream>

void kernel_log(hb_tensor_t* arg) {
  std::cout << "kernel log: " << std::endl;
  for(int i=0; i<arg->N; ++i) {
    std::cout << ((float*) ((intptr_t) arg->data))[i] << " ";
  }
  std::cout << std::endl;
}

void kernel_log(float* arg) {
  std::cout << "kernel log: " << *arg << std::endl;
}
