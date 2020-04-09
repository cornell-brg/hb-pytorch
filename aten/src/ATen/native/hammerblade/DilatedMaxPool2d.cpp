#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/Pool.h>

namespace at {
namespace native {
namespace { // anonumous

void offload_max_pool2d_with_indices(
    Tensor& output, const Tensor& input,
    Tensor& indices, int kH, int kW,
    int dH, int dW, int padH, int padW,
    int dilationH, int dilationW) {
  TORCH_CHECK(dilationH == 1 && dilationW == 1, 
              "Dilated max_pool2d is not by HB yet.");

  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(input, device_ptrs));
  device_args.push_back(create_device_tensor(indices, device_ptrs));
  device_args.push_back(create_device_scalar(kH));
  device_args.push_back(create_device_scalar(kW));
  device_args.push_back(create_device_scalar(dH));
  device_args.push_back(create_device_scalar(dW));
  device_args.push_back(create_device_scalar(padH));
  device_args.push_back(create_device_scalar(padW));
  device_args.push_back(create_device_scalar(dilationH));
  device_args.push_back(create_device_scalar(dilationW));

  c10::hammerblade::offload_kernel(
      "tensorlib_max_pool2d", device_args);
  cleanup_device(device_args, device_ptrs);
}

void max_pool2d_with_indices_out_hb_template(
           Tensor& output,
           Tensor& indices,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameHB("max_pool2d_with_indices_out_hb",
                  {output_arg, indices_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous();

  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  const int count = safe_downcast<int, int64_t>(output.numel());

  AT_DISPATCH_FLOAT_TYPE_ONLY(input.scalar_type(),
    "max_pool2d_with_indices_out_hb_frame",
    [&] {
      offload_max_pool2d_with_indices(
        output, input, indices,
        kH, kW, dH, dW, padH, padW,
        dilationH, dilationW);
    }
  );

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

void max_pool2d_with_indices_backward_out_hb_template(
           Tensor& gradInput,
           const Tensor& gradOutput_,
           const Tensor& input_,
           const Tensor& indices,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameHB("max_pool2d_with_indices_out_hb",
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");
  const Tensor input = input_.contiguous();

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    input_,
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  const Tensor gradOutput = gradOutput_.contiguous();

  gradInput.resize_as_(input);
  gradInput.contiguous();

  AT_DISPATCH_FLOAT_TYPE_ONLY(input.scalar_type(),
    "max_pool2d_with_indices_backward",
    [&] {
      std::vector<eva_t> device_args;
      std::vector<eva_t> device_ptrs;
      device_args.push_back(create_device_tensor(gradInput, device_ptrs));
      device_args.push_back(create_device_tensor(gradOutput, device_ptrs));
      device_args.push_back(create_device_tensor(indices, device_ptrs));
      device_args.push_back(create_device_tensor(input_, device_ptrs));
      device_args.push_back(create_device_scalar(kH));
      device_args.push_back(create_device_scalar(kW));
      device_args.push_back(create_device_scalar(dH));
      device_args.push_back(create_device_scalar(dW));
      device_args.push_back(create_device_scalar(padH));
      device_args.push_back(create_device_scalar(padW));

      c10::hammerblade::offload_kernel(
          "tensorlib_max_pool2d_backward", device_args);
      cleanup_device(device_args, device_ptrs);
    }
  );
}

} // anonymous

std::tuple<Tensor, Tensor> max_pool2d_with_indices_hb(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kInt));
  max_pool2d_with_indices_out_hb_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor max_pool2d_with_indices_backward_hb(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  auto gradInput = at::zeros(input.sizes(), input.options());
  max_pool2d_with_indices_backward_out_hb_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

}} // at::native
