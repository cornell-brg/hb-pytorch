#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>
#include <ATen/native/UpSample.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

static void upsample_nearest1d_out_hb_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_width = input_.size(2);


  upsample_1d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_width,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_width});
  output.zero_();

  AT_ASSERT(input_width > 0 && output_width > 0);

  int32_t nbatch_32 = (int32_t)nbatch;
  int32_t channels_32 = (int32_t)channels;
  int32_t input_width_32 = (int32_t)input_width;
  int32_t output_width_32 = (int32_t)output_width;

  std::vector<eva_t> scalar_args;
  std::vector<Tensor> tensor_args;
  tensor_args.push_back(output);
  tensor_args.push_back(input_);
  scalar_args.push_back(create_device_scalar(input_width_32));
  scalar_args.push_back(create_device_scalar(output_width_32));
  scalar_args.push_back(create_device_scalar(nbatch_32));
  scalar_args.push_back(create_device_scalar(channels_32));

  offload_tensor_scalar_impl(tensor_args,scalar_args,"tensorlib_upsample_nearest1d");
}

Tensor& upsample_nearest1d_out_hb(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest1d_out_hb_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest1d_hb(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  upsample_nearest1d_out_hb(output, input, output_size);
  return output;
}

}} // namespace at::native
