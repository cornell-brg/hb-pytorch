#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

namespace { // anonymous

// Offload routine convolution forward pass
void offload_convolution_forward(Tensor& output, const Tensor& input,
    const Tensor& weight, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {

  // Dimension check
  TORCH_CHECK(output.dim() == 4, "Only 2d convolution supported now.");

  // Dilation check
  bool dilation_check = true;
  for(auto d : dilation) {
    if(d != 1) {
      TORCH_WARN("dilation[i] = ", d);
      dilation_check = false;
      break;
    }
  }
  TORCH_CHECK(dilation_check,
        "dilation = ", dilation,
        " is not supported by HB yet.",
        " Make sure dilation is all ones.");

  // Groups check
  TORCH_CHECK(groups == 1,
      "Grouped convolution not supported by HB yet."
      " Make sure groups = 1.");

  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(input, device_ptrs));
  device_args.push_back(create_device_tensor(weight, device_ptrs));
  device_args.push_back(create_device_vector(padding, true, device_ptrs));
  device_args.push_back(create_device_vector(stride, true, device_ptrs));

  c10::hammerblade::offload_kernel(
      "tensorlib_convolution_forward", device_args);
  cleanup_device(device_args, device_ptrs);
}

// Offload routine for covolution bias addition
void offload_convolution_add_bias(const Tensor& output, const Tensor& bias) {
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(bias, device_ptrs));

  c10::hammerblade::offload_kernel(
      "tensorlib_convolution_add_bias", device_args);
  cleanup_device(device_args, device_ptrs);
}

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

static std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups
) {
  auto dim = input_size.size();
  std::vector<int64_t> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + 2 * padding[d - 2] - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  TORCH_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
           "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    AT_ERROR(ss.str());
  }
}

static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

Tensor hb_convolution_forward(
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

  offload_convolution_forward(
      output_t, *input, *weight,
      padding, stride, dilation, groups);

  return *output;
}

// In-place!
void hb_convolution_add_bias_(CheckedFrom c, const TensorArg& output, 
                              const TensorArg& bias) {
  checkAllSameType(c, {output, bias});
  checkAllSameHB(c, {output, bias});
  checkSize(c, bias, { output->size(output_channels_dim) });

  if (output.tensor.numel() == 0) {
    return;
  }

  offload_convolution_add_bias(*output, *bias);
}


} // anonymous namespace

Tensor hb_convolution_transpose(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef output_padding,
    IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {

  TORCH_CHECK(false, "hb_convolution_transpose: not yet implemented!");
  return at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kFloat));
}

Tensor hb_convolution(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride, 
    IntArrayRef dilation, int64_t groups) {
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "hb_convolution";
  auto output_t = hb_convolution_forward(
    c, input, weight, padding, stride, dilation, groups);
  if (bias->defined()) {
    hb_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

Tensor hb_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("hb_convolution_backward_input: not implemented yet.");
}

Tensor hb_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("hb_convolution_backward_weight: not implemented yet.");
}

Tensor hb_convolution_backward_bias(
    const at::Tensor& grad_output) {
  AT_ERROR("hb_convolution_backward_bias: not implemented yet.");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> hb_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output,
    const at::Tensor& weight, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups,
    std::array<bool,3> output_mask) {
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty(input.sizes(), input.options());
    }
    if (output_mask[1]) {
      grad_weight = at::zeros(weight.sizes(), weight.options());
    }
    if (output_mask[2]) {
      grad_bias = at::zeros({grad_output.size(1)}, grad_output.options());
    }
  } else {
    if (output_mask[0]) {
      grad_input = at::hb_convolution_backward_input(
          input.sizes(), grad_output, weight, padding,
          stride, dilation, groups);
    }
    if (output_mask[1]) {
      grad_weight = at::hb_convolution_backward_weight(
          weight.sizes(), grad_output, input, padding,
          stride, dilation, groups);
    }
    if (output_mask[2]) {
      grad_bias = at::hb_convolution_backward_bias(grad_output);
    }
  }

  return std::tuple<Tensor,Tensor,Tensor>{
    grad_input, grad_weight, grad_bias};
}

}} // namespace at::native
