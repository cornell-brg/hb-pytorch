#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor and_kernel_hb(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.numel() == other.numel(), "The size of two tensors should match.");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), "two inputs should have the same type");
  TORCH_CHECK(other.scalar_type() == kInt || other.scalar_type() == kBool, "HammerBlade and is implemented for Int and Bool only");
  Tensor result = at::empty_like(self, self.options());
  if (self.scalar_type() ==kInt) {
    hb_offload_kernel(result, self, other, "tensorlib_and_int");
  }
  else {
    hb_offload_kernel(result, self, other, "tensorlib_and_bool");
  }
  
  return result;
}

}} // namespace at::native
