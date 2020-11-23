#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

namespace {
  void _hardtanh_hb(Tensor& out, Tensor const& self,
                   Scalar min, Scalar max) {
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;
    device_args.push_back(create_device_tensor(out, device_ptrs));
    device_args.push_back(create_device_tensor(self, device_ptrs));
    device_args.push_back(create_device_scalar(min.to<float>()));
    device_args.push_back(create_device_scalar(max.to<float>()));
    c10::hammerblade::offload_kernel(
        "tensorlib_hardtanh", device_args);
    cleanup_device(device_args, device_ptrs);
  }
}

Tensor hardtanh_hb(Tensor const& self, Scalar min, Scalar max) {
  auto out = at::empty(self.sizes(), self.options());
  _hardtanh_hb(out, self, min, max);
  return out;
}

Tensor hardtanh_hb_(Tensor& self, Scalar min, Scalar max) {
  _hardtanh_hb(self, self, min, max);
  return self;
}
}} // at::native
