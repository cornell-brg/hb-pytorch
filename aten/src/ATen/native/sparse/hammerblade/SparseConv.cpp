#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native{
/*
Tensor hb_sparse_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  checkAllSameType(c, {input, weight});
  checkAllSameHB(c, {input, weight});

  auto output_t = at::empty(
                    conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation, groups),
                    input->options());

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };

  convolution_shape_check(c, input, weight, output, padding, stride,
             dilation, groups);

  Tensor weight_contig = weight->contiguous();

  hb_convolution_arg_check(output->dim(), dilation, groups);
  
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output_t, device_ptrs));
  device_args.push_back(create_device_tensor(*input, device_ptrs));
  device_args.push_back(create_device_tensor(*weight, device_ptrs));
  device_args.push_back(create_device_vector(padding, true, device_ptrs));
  device_args.push_back(create_device_vector(stride, true, device_ptrs));
  
  c10::hammerblade::offload_kernel(
      "tensorlib_sparse_convolution_forward", device_args);
  cleanup_device(device_args, device_ptrs);
  
  return *output;
}
*/
Tensor hb_sparse_convolution(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {
/*
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "hb_convolution";

  auto output_t = hb_sparse_convolution_forward(
    c, input, weight, padding, stride, dilation, groups);


  if (bias->defined()) {
    hb_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
*/
  Tensor output_t;
  return output_t;
}

}}
