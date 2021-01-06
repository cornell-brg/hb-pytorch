#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

/*
template <typename scalar_t>
static void upsample_nearest1d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_width,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels) {
  const float scale = (float)input_width / (float)output_width;
  channels = channels * nbatch;

  // special case: just copy
  if (input_width == output_width) {
    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 = w2;
      const scalar_t* pos1 = &idata[w1];
      scalar_t* pos2 = &odata[w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    return;
  }

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const scalar_t src_x =
        nearest_neighbor_compute_source_index(scale, w2, input_width);
    const int64_t w1 = src_x;
    const scalar_t* pos1 = &idata[w1];
    scalar_t* pos2 = &odata[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
}
*/

template <typename scalar_t>
static void upsample_nearest1d_backward_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_width,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels) {
  const float scale = (float)input_width / (float)output_width;
  channels = channels * nbatch;

  // special case: same-size matching grids
  if (input_width == output_width) {
    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 = w2;
      scalar_t* pos1 = &idata[w1];
      const scalar_t* pos2 = &odata[w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    return;
  }

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const int64_t w1 =
        nearest_neighbor_compute_source_index(scale, w2, input_width);

    scalar_t* pos1 = &idata[w1];
    const scalar_t* pos2 = &odata[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
}

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

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

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
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();

        upsample_nearest1d_backward_out_frame<scalar_t>(
            odata,
            idata,
            input_width,
            output_width,
            nbatch,
            channels);
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

