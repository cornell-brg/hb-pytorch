#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/ExpandUtils.h>

#include <iostream>

namespace at { namespace native {

Tensor dummy_hb(
    const Tensor& self,
    const Tensor& other
) {

  TORCH_CHECK(self.dim() == other.dim(), "Shapes of the two input tensors have to be the same");

  // Offloading code here

  auto result = at::empty({self.size(0)}, self.options());

  hb_offload_kernel(result, self, other, "tensorlib_dummy");

  return result;
}

}} // namespace at::native
