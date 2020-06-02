#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor vincr_hb(const Tensor& self) {

  // Create an output tensor that has the same shape as self
  auto result = at::empty_like(self, self.options());

  // Call HB device kernel tensorlib_vincr
  hb_offload_kernel(result, self, "tensorlib_vincr");

  return result;
}

}} // namespace at::native
