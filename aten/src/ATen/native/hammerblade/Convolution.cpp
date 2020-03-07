#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>


namespace at {
namespace native {

Tensor& hb_convolution_nogroup(
    const Tensor& input, const Tensor& weight, 
    const Tensor& bias, IntArrayRef stride, 
    IntArrayRef padding, IntArrayRef dilation, 
    bool transposed, IntArrayRef output_padding){

  TORCH_CHECK(false, "hb_convolution_nogroup: not yet implemented!");

  return input;
}

}} // namespace at::native
