#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor mm_hb(const Tensor& input, const Tensor& mat2) {
  TORCH_CHECK(input.scalar_type() == ScalarType::Float, "HammerBlade addmm is implemented for Float only");
  TORCH_CHECK(mat2.scalar_type() == ScalarType::Float, "HammerBlade addmm is implemented for Float only");
  TORCH_CHECK(input.dim() == 2 && mat2.dim() == 2, "2D matrices expected, got ", input.dim(), " and ", mat2.dim(), " tensors");
  TORCH_CHECK(input.size(1) == mat2.size(0), "Argument #3: Expected dim 0 size ", input.size(1), ", got ", mat2.size(0));

  Tensor result = at::empty({input.size(0), mat2.size(1)}, input.options());

  hb_offload_kernel(result, input, mat2, "tensorlib_mm");

  return result;
}

}} // namespace at::native

