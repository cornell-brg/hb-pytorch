#include <ATen/ATen.h>

namespace at {
namespace native {

namespace { // anonymous

Tensor hb_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, 
    int64_t groups)
{
  checkAllSameType(c, {input, weight});
  checkAllSameHB(c, {input, weight});

  TORCH_CHECK(false, "hb_convolution_forward: not implemeted yet!");
  return at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kFloat));
}

} // anonymous namespace

Tensor hb_convolution_transpose(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef output_padding,
    IntArrayRef stride, IntArrayRef dilation,
    int64_t groups){

  TORCH_CHECK(false, "hb_convolution_transpose: not yet implemented!");
  return at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kFloat));
}

Tensor hb_convolution(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride, 
    IntArrayRef dilation, int64_t groups){
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "hb_convolution";
  auto output_t = hb_convolution_forward(
    c, input, weight, padding, stride, dilation, groups);
  if (bias->defined()) {
    // TODO: HB_TODO_VB hb_convolution_add_bias_
    // hb_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
    TORCH_CHECK(false, "hb_convolution: adding bias not implemented yet!");
  }
  return output_t;
}

}} // namespace at::native
