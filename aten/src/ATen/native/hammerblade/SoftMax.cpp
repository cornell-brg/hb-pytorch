#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor log_softmax_hb(const Tensor& input_, const int64_t dim_,
                      const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on HB");

  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input,
                                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());

  if (input.numel() == 0) {
    return output;
  }

  if (input.dim() == 0)
    input = input.view(1);

  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");

  AT_DISPATCH_FLOAT_TYPE_ONLY(
    input.scalar_type(), "log_softmax_hb",
    [&] {
      std::vector<eva_t> device_args;
      std::vector<eva_t> device_ptrs;
      device_args.push_back(create_device_tensor(output, device_ptrs));
      device_args.push_back(create_device_tensor(input, device_ptrs));
      device_args.push_back(create_device_scalar(dim));

      c10::hammerblade::offload_kernel(
          "tensorlib_log_softmax", device_args);
      cleanup_device(device_args, device_ptrs);
    });

  return output;
}

}} // at::native
