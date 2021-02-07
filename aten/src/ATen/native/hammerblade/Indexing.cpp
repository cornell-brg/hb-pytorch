#include <ATen/native/Indexing.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

namespace at { namespace native {
namespace {
void index_kernel_hb(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {

//  std::cout << "Enter index_kernel_hb function." << std::endl;
  auto strides = iter.get_strides();
  while (strides.size() < 2 * iter.ntensors()) {
    strides.push_back(0);
  }

  for(int i = 0; i < strides.size(); i++) {
    strides[i] = strides[i] / 4;
//    std::cout << "  strides[" << i << "] is: " << strides[i] << std::endl;
  }

  IntArrayRef kernel_strides = IntArrayRef(strides);
  IntArrayRef shapes = iter.shape();
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  for(int i = 0; i < iter.ntensors(); i++) { 
    Tensor& t = iter.tensor(i);
    if(i > iter.noutputs() && t.dtype() == at::kLong) {
      std::cout << "Convert data type" << std::endl;
      t.to(at::kInt);
    }
    device_args.push_back(create_device_tensor(t, device_ptrs));
//    std::cout << "  Tensor number of dimension is: " << t.ndimension() << std::endl;
//    std::cout << "  Tensor number of strides is: " << t.strides().size() << std::endl;
  }
  device_args.push_back(create_device_vector(shapes, true, device_ptrs));
  device_args.push_back(create_device_vector(kernel_strides, true, device_ptrs));
  device_args.push_back(create_device_vector(index_size, true, device_ptrs));
  device_args.push_back(create_device_vector(index_stride, true, device_ptrs));

  if(iter.ntensors() == 3) {
    c10::hammerblade::offload_kernel(
        "tensorlib_indexing_1d", device_args);
  } else if(iter.ntensors() == 4) {
    c10::hammerblade::offload_kernel(
        "tensorlib_indexing_2d", device_args);
  } else {
    AT_ERROR("Currently, we only support the indexing of 2D HB Tensor !");
  }
  
  cleanup_device(device_args, device_ptrs);
}

REGISTER_HAMMERBLADE_DISPATCH(index_stub, &index_kernel_hb);

}
}}

