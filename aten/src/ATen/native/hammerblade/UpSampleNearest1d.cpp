#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void upsample_nearest1d_out_hb_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  int32_t output_width = safe_downcast<int32_t, int64_t>(output_size[0]);
  int32_t nbatch       = safe_downcast<int32_t, int64_t>(input_.size(0));
  int32_t channels     = safe_downcast<int32_t, int64_t>(input_.size(1));
  int32_t input_width  = safe_downcast<int32_t, int64_t>(input_.size(2));

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

  AT_DISPATCH_FLOAT_TYPE_ONLY(
    input.scalar_type(), "upsample_nearest1d", [&] {
      hb_offload_kernel(
          output,
          input,
          input_width,
          output_width,
          nbatch,
          channels,
          "tensorlib_upsample_nearest1d");
  });
}

static void upsample_nearest1d_backward_out_hb_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int32_t output_width = safe_downcast<int32_t, int64_t>(output_size[0]);
  int32_t nbatch       = safe_downcast<int32_t, int64_t>(input_size[0]);
  int32_t channels     = safe_downcast<int32_t, int64_t>(input_size[1]);
  int32_t input_width  = safe_downcast<int32_t, int64_t>(input_size[2]);

  upsample_1d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_width,
      output_width);

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  AT_DISPATCH_FLOAT_TYPE_ONLY(
      grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
        hb_offload_kernel(
            grad_output,
            grad_input,
            input_width,
            output_width,
            nbatch,
            channels,
            "tensorlib_upsample_nearest1d_back");
      });
}
} // namespace
// forward

Tensor& upsample_nearest1d_out_hb(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest1d_out_hb_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest1d_hb(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  upsample_nearest1d_out_hb_template(output, input, output_size);
  return output;
}

// backward

Tensor& upsample_nearest1d_backward_out_hb(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  upsample_nearest1d_backward_out_hb_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest1d_backward_hb(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_nearest1d_backward_out_hb_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

}} // namespace at::native

