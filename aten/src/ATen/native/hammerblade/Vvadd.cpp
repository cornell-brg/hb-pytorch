#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor vvadd_hb(const Tensor& self, const Tensor& other) {

  // Create an output tensor that has the same shape as self
  auto result = at::empty_like(self, self.options());

  // Tutorial TODO:
  // Call HB device kernel tensorlib_vvadd
  hb_offload_kernel(result, self, other, "tensorlib_vvadd");

  return result;
}

}} // namespace at::native
