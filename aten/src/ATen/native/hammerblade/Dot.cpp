#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor dot_hb(const Tensor& self, const Tensor& other) {
#ifdef BUILD_NAMEDTENSOR
  at::NoNamesGuard guard;
#endif
  if ( (self.dim() != 1) || (other.dim() != 1) ) {
    AT_ERROR("1D tensors expected, got ",self.dim(), "D, ", other.dim(), "D tensors");
  }
  if ( self.numel() != other.numel() ) {
    AT_ERROR("Tensor size mismatch, got src=", self.numel(), ", dst=", other.numel());
  }
  if ( (self.scalar_type() != ScalarType::Float) || (other.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade dot is implemented for Float only");
  }
  auto sum = at::empty({}, other.options());

  HB_OFFLOAD_TENSOR_KERNEL(sum, self, other, "tensorlib_dot");

  return sum;
}

}} // namespace at::native
