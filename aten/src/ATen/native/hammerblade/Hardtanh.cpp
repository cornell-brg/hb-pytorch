#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {
  Tensor hardtanh_hb(Tensor const& self, Scalar min, Scalar max) {
    auto out = at::empty(self.sizes(), self.options());

    std::vector<Tensor> args;
    args.push_back(out);
    args.push_back(self);
    std::vector<eva_t> scalars;
    scalars.push_back(create_device_scalar(min));
    scalars.push_back(create_device_scalar(max));
    offload_tensor_scalar_impl(args, scalars, "tensorlib_hardtanh");
    return out;
  }
}} // at::native
