#include <tuple>
#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

std::tuple<Tensor, Tensor> lu_hb(const Tensor& self) {

  TORCH_CHECK(self.scalar_type() == ScalarType::Float, "HammerBlade lu is implemented for Float only");

  // TODO: implement LU for batch of matrices and modify this check
  TORCH_CHECK(self.dim() == 2, "2D matrices expected, got ", self.dim(), " tensor");

  // TODO: implement LU for non-square matrices and remove this check
  TORCH_CHECK(self.size(0) == self.size(1), "Square matrices expected, got ", self.size(0), " by ", self.size(1), " tensor");

  //Tensor factorization = at::empty({self.size(0), self.size(1)}, self.options()); // mXn matrix, same as self
  Tensor factorization = at::clone(self); // mXn matrix, same as self
  Tensor pivots = at::empty({self.size(0)}, self.options()); // mX1 vector

  hb_offload_kernel(factorization, pivots, "tensorlib_lu");

  return std::tie(factorization, pivots);
}

}} // namespace at::native
