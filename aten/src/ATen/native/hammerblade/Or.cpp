#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor or_kernel_hb(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.numel() == other.numel(), "The size of two tensors should match.");
  // TORCH_CHECK(self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool, "HammerBlade or is implemented for Int and Bool only");
  // TORCH_CHECK(other.scalar_type() == ScalarType::Int || other.scalar_type() == ScalarType::Bool, "HammerBlade or is implemented for Int and Bool only");
  Tensor result = at::empty_like(self, self.options());
  hb_offload_kernel(result, self, other, "tensorlib_or");
  return result;
}

}} // namespace at::native
