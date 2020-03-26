#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

//--------------------------------------
// Forward
//--------------------------------------

Tensor log_softmax_hb(const Tensor& input_, const int64_t dim_,
                      const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on HB");

  auto input = input_.contiguous();
  Tensor output = at::empty(input.sizes(), input.options());
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  int32_t dim_i32 = safe_downcast<int32_t, int64_t>(dim);

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
      hb_offload_kernel(output, input, dim_i32,
                        "tensorlib_log_softmax");
    });

  return output;
}


//--------------------------------------
// Backward
//--------------------------------------

Tensor log_softmax_backward_hb(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize("log_softmax_backward", grad_arg, output_arg);
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  int32_t dim_i32 = safe_downcast<int32_t, int64_t>(dim);
  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  Tensor grad_input = at::native::empty_like(grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (output.numel() == 0) {
    return grad_input;
  }
  if (grad.dim() == 0)
    grad = grad.view(1);
  if (output.dim() == 0)
    output = output.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");

  AT_DISPATCH_FLOAT_TYPE_ONLY(
      grad.scalar_type(), "log_softmax_backward_hb",
      [&] {
        hb_offload_kernel(grad_input, grad, output,
                          dim_i32,
                          "tensorlib_log_softmax_backward");
      });

  return grad_input;
}

}} // at::native
