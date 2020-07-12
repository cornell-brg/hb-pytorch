#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> batch_norm_hb(
    const Tensor& self, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool train, double momentum, double eps) {
  checkBackend("batch_norm_hb",
               {self, weight, bias, running_mean, running_var},
               Backend::HammerBlade);

  TORCH_CHECK(false, "HB BatchNorm kernel called! (unimplemented)");

  //return AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "batch_norm", [&] {
  //    if (!train) {
  //      return batch_norm_cpu_transform_input_template<scalar_t>(
  //          self, weight, bias, {}, {}, running_mean, running_var, train, eps);
  //    } else {
  //      auto save_stats = batch_norm_cpu_update_stats_template<scalar_t, InvStd>(
  //          self, running_mean, running_var, momentum, eps);
  //      return batch_norm_cpu_transform_input_template<scalar_t>(
  //          self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats),
  //          running_mean, running_var, train, eps);
  //    }
  //  });
}

}} // namespace at::native
