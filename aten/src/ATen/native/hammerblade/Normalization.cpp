#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

std::tuple<Tensor,Tensor,Tensor> batch_norm_hb_transform_input(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean /* optional */,
    const Tensor& save_invstd /* optional */,
    const Tensor& running_mean /* optional */,
    const Tensor& running_var /* optional */,
    bool train, double eps) {
  TORCH_CHECK(false, "HB BatchNorm transform input unimplemented");
  return std::make_tuple(input, save_mean, save_invstd);
}

std::tuple<Tensor,Tensor> batch_norm_hb_update_stats(
    const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
    double momentum, double eps) {
  TORCH_CHECK(false, "HB BatchNorm update stats unimplemented");

  int64_t n_input = input.size(1);
  int64_t n = input.numel() / n_input;

  Tensor save_mean = at::empty({n_input}, input.options());
  Tensor save_var_transform = at::empty({n_input}, input.options());
  return std::make_tuple(save_mean, save_var_transform);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_hb(
    const Tensor& self, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool train, double momentum, double eps) {
  checkBackend("batch_norm_hb",
               {self, weight, bias, running_mean, running_var},
               Backend::HammerBlade);

  return AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "batch_norm", [&] {
      if (!train) {
        return batch_norm_hb_transform_input(
            self, weight, bias, {}, {}, running_mean, running_var, train, eps);
      } else {
        auto save_stats = batch_norm_hb_update_stats(
            self, running_mean, running_var, momentum, eps);
        return batch_norm_hb_transform_input(
            self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats),
            running_mean, running_var, train, eps);
      }
    });
}

}} // namespace at::native
