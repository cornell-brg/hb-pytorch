#include <ATen/ATen.h>
#include <iostream>

namespace at {
namespace native {

Tensor add_amc_cpu(const Tensor& self, const Tensor& other) {
  // allocate output tensor
  Tensor result = at::empty_like(self, self.options());

  // get raw pointer, assuming uint32
  size_t size = self.sizes()[0];
  int * src0_arr = (int *) self.data_ptr<int>();
  int * src1_arr = (int *) other.data_ptr<int>();
  int * dest_arr = (int *) result.data_ptr<int>();

  // vvadd -- replace with RISCV code when ready for gem5
  std::cout << "faking AMC vector add ..." << std::endl;
  for (int i = 0; i < size; i++) {
    dest_arr[i] = src0_arr[i] + src1_arr[i];
  }

  // return result
  return result;
}

}} // at::native
