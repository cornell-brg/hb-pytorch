#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/ExpandUtils.h>

#include <iostream>

namespace at { namespace native {

Tensor dummy_hb(
    const Tensor& self
) {

  if ((self.scalar_type() != ScalarType::Float)) {
    AT_ERROR("HammerBlade dummy is implemented for Float only");
  }
  auto result = at::empty({self.size(0)}, self.options());

  hb_offload_kernel(result, self, "tensorlib_dummy");

  return result;
}

}} // namespace at::native
