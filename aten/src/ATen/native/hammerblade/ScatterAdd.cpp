#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

#include <iostream>

namespace at { namespace native {
Tensor& scatter_add_hb_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    
  TORCH_CHECK_INDEX(index.dim() == self.dim(), "Index tensor must have same dimensions as output tensor");
  TORCH_CHECK_INDEX(self.dim() == src.dim(), "Input tensor must have same dimensions as output tensor");
  Tensor index_int;
  if(index.dtype() == kLong) {
    index_int = index.to(kInt);
  }
  hb_offload_kernel(self, index_int, src, dim, "tensorlib_scatter_add");
  return self;  
}

}}

