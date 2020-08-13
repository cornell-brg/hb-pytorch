#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor vvadd_hb(const Tensor& self, const Tensor& other) {

  TORCH_CHECK(self.scalar_type() == ScalarType::Float, "HammerBlade addmm is implemented for Float only");
  TORCH_CHECK(other.scalar_type() == ScalarType::Float, "HammerBlade addmm is implemented for Float only");
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "2D matrices expected, got ", self.dim(), " and ", other.dim(), " tensors");
  TORCH_CHECK(self.size(1) == other.size(0), "Argument #3: Expected dim 0 size ", self.size(1), ", got ", other.size(0));

  Tensor result = at::empty({self.size(0), other.size(1)}, self.options());

  hb_offload_kernel(result, self, other, "tensorlib_mm");

  return result;
}

}} // namespace at::native
