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
  hb_offload_kernel("tensorlib_nearest1d_hb");
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_nearest1d", [&] {
  //   auto* idata = input.data_ptr();
  //   auto* odata = output.data_ptr();

  //   std::vector<eva_t> device_args;
  //   device_args.push_back(create_device_scalar(odata));
  //   device_args.push_back(create_device_scalar(idata));
  //   device_args.push_back(create_device_scalar(input_width));
  //   device_args.push_back(create_device_scalar(output_width));
  //   device_args.push_back(create_device_scalar(nbatch));
  //   device_args.push_back(create_device_scalar(channels));

  //   // c10::hammerblade::offload_kernel("tensorlib_nearest1d_hb");
  // //   // , device_args);
    
  
  // });
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
  // printf("in upsample nearest1d hb");
  upsample_nearest1d_out_hb(output, input, output_size);
  return output;
}

}} // namespace at::native
