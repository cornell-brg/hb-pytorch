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
  TORCH_CHECK(input.dim() == 4,
              "Only 2d BatchNorm is implemented for HB backend.");
  Tensor output = at::empty(input.sizes(), input.options());

  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(input, device_ptrs));
  device_args.push_back(create_device_tensor(weight, device_ptrs));
  device_args.push_back(create_device_tensor(bias, device_ptrs));
  device_args.push_back(create_device_tensor(save_mean, device_ptrs));
  device_args.push_back(create_device_tensor(save_invstd, device_ptrs));
  device_args.push_back(create_device_tensor(running_mean, device_ptrs));
  device_args.push_back(create_device_tensor(running_var, device_ptrs));
  device_args.push_back(create_device_scalar((int) train));
  device_args.push_back(create_device_scalar((float) eps));
  c10::hammerblade::offload_kernel("tensorlib_batch_norm2d_transform_input",
                                   device_args);
  cleanup_device(device_args, device_ptrs);

  return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor,Tensor> batch_norm_hb_update_stats(
    const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
    double momentum, double eps) {
  TORCH_CHECK(input.dim() == 4,
              "Only 2d BatchNorm is implemented for HB backend.");

  int64_t n_input = input.size(1);

  Tensor save_mean = at::empty({n_input}, input.options());
  Tensor save_invstd = at::empty({n_input}, input.options());

  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(save_mean, device_ptrs));
  device_args.push_back(create_device_tensor(save_invstd, device_ptrs));
  device_args.push_back(create_device_tensor(input, device_ptrs));
  device_args.push_back(create_device_tensor(running_mean, device_ptrs));
  device_args.push_back(create_device_tensor(running_var, device_ptrs));
  device_args.push_back(create_device_scalar((float) momentum));
  device_args.push_back(create_device_scalar((float) eps));
  c10::hammerblade::offload_kernel("tensorlib_batch_norm2d_update_stats",
                                   device_args);
  cleanup_device(device_args, device_ptrs);

  return std::make_tuple(save_mean, save_invstd);
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

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_hb(
    const Tensor& grad_out, const Tensor& self, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps, std::array<bool,3> grad_input_mask) {
  TORCH_CHECK(self.dim() == 4,
              "Only 2d BatchNorm is implemented for HB backend.");

  return AT_DISPATCH_FLOAT_TYPE_ONLY(
      self.scalar_type(), "batch_norm_backward_hb", [&] {
      Tensor grad_input;
      Tensor grad_weight;
      Tensor grad_bias;
      if (grad_input_mask[0]) {
        grad_input = at::empty(self.sizes(), self.options());
      }
      if (grad_input_mask[1]) {
        grad_weight = at::empty(weight.sizes(), weight.options());
      }
      if (grad_input_mask[2]) {
        grad_bias = at::empty(weight.sizes(), weight.options());
      }

      std::vector<eva_t> device_args;
      std::vector<eva_t> device_ptrs;
      device_args.push_back(create_device_tensor(grad_input, device_ptrs));
      device_args.push_back(create_device_tensor(grad_weight, device_ptrs));
      device_args.push_back(create_device_tensor(grad_bias, device_ptrs));
      device_args.push_back(create_device_tensor(grad_out, device_ptrs));
      device_args.push_back(create_device_tensor(self, device_ptrs));
      device_args.push_back(create_device_tensor(weight, device_ptrs));
      device_args.push_back(create_device_tensor(save_mean, device_ptrs));
      device_args.push_back(create_device_tensor(save_invstd, device_ptrs));
      device_args.push_back(create_device_tensor(running_mean, device_ptrs));
      device_args.push_back(create_device_tensor(running_var, device_ptrs));
      device_args.push_back(create_device_scalar((int) train));
      device_args.push_back(create_device_scalar((float) eps));
      c10::hammerblade::offload_kernel("tensorlib_batch_norm2d_backward",
                                       device_args);
      cleanup_device(device_args, device_ptrs);

      return std::make_tuple(grad_input, grad_weight, grad_bias);
    });
}

}} // namespace at::native
