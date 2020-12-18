#include <tuple>
#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor cholesky_hb(const Tensor& self) {

  TORCH_CHECK(self.scalar_type() == ScalarType::Float, "HammerBlade Cholesky is implemented for Float only");
  TORCH_CHECK(self.dim() == 2, "2D matrices expected, got ", self.dim(), " tensor");
  TORCH_CHECK(self.size(0) == self.size(1), "Square matrices expected, got ", self.size(0), " by ", self.size(1), " tensor");

  Tensor factorization = at::clone(self);

  hb_offload_kernel(factorization, "tensorlib_cholesky");

  return factorization;
}

}} // namespace at::native
