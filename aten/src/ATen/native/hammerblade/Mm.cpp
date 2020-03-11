#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor mm_hb(const Tensor& input, const Tensor& mat2) {
  Tensor t = at::zeros({}, mat2.options());
  return at::addmm(t, input, mat2, 0, 1); // redispatch!
}

Tensor& mm_out_hb(Tensor& result, const Tensor& input, const Tensor& mat2) {
  Tensor t = at::zeros({}, mat2.options());
  return at::addmm_out(result, t, input, mat2, 0, 1); // redispatch!
}

}} // namespace at::native

