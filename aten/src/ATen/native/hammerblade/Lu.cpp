#include <tuple>
#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> lu_hb(const Tensor& self, const bool pivot, const bool get_infos) {

  TORCH_CHECK(self.scalar_type() == ScalarType::Float, "HammerBlade lu is implemented for Float only");

  // TODO: implement LU for batch of matrices and modify this check
  TORCH_CHECK(self.dim() == 2, "2D matrices expected, got ", self.dim(), " tensor");

  // TODO: implement LU for non-square matrices and remove this check
  TORCH_CHECK(self.size(0) == self.size(1), "Square matrices expected, got ", self.size(0), " by ", self.size(1), " tensor");

  std::tuple<Tensor, Tensor, Tensor> result;
  Tensor factorization = at::empty({self.size(0), self.size(1)}, self.options()); // mXn matrix, same as self
  Tensor pivots = at::empty({self.size(0)}, self.options()); // mX1 vector
  Tensor infos = at::empty({1}, self.options()); // just one matrix in the batch right now

  hb_offload_kernel(factorization, pivots, infos, self, pivot, get_infos, "tensorlib_lu");

  //if (get_infos) return std::tie(factorization, pivots, infos);
  //return std::tie(factorization, pivots);
  return std::tie(factorization, pivots, infos);
}

}} // namespace at::native
