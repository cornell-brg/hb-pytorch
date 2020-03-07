#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>


namespace at {
namespace native {

at::Tensor hb_convolution_transpose(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef output_padding,
    IntArrayRef stride, IntArrayRef dilation,
    int64_t groups){

  TORCH_CHECK(false, "hb_convolution_transpose: not yet implemented!");
  return at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kFloat));
}

at::Tensor hb_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, 
    IntArrayRef dilation, int64_t groups){

  TORCH_CHECK(false, "hb_convolution_transpose: not yet implemented!");
  return at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kFloat));
}

}} // namespace at::native
