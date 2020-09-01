#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor hb_convolution(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride, 
    IntArrayRef dilation, int64_t groups) {
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "hb_convolution";
  return at::empty({1}, input_t->options());
}

}} // at::native
