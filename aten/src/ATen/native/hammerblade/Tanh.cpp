#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& tanh_(Tensor& self) {

  AT_DISPATCH_FLOAT_TYPE_ONLY(
    self.scalar_type(), "tanh_",
    [&] {
      hb_offload_kernel(self, "tensorlib_tanh");
    });
  return self;
}

Tensor& tanh_out_(Tensor& out, const Tensor& self){

  AT_DISPATCH_FLOAT_TYPE_ONLY(
    self.scalar_type(), "tanh_out_",
    [&] {
      hb_offload_kernel(self, out, "tensorlib_tanh_out");
    });
  return out;
}
}}  // namespace at::native


