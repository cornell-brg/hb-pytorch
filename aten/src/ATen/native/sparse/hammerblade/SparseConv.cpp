#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native{

using namespace at::sparse;

IntTensor _to_csr_int( const IntTensor& rowIndices, int64_t dim, int64_t nnz);

void hb_sparse_convolution_arg_check(int64_t output_dims, IntArrayRef dilation, int64_t groups) {
  // Dimension check
  TORCH_CHECK(output_dims == 4, "Only 2d convolution supported now.");
  
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
        "dilation = ", dilation, " is not supported by HB yet.", " Make sure dilation is all ones.");
  
  // Groups check
  TORCH_CHECK(groups == 1, "Grouped convolution not supported by HB yet.", " Make sure groups = 1.");
}

constexpr int dense_input_batch_size_dim = 0;
constexpr int dense_input_channels_dim = 1;
constexpr int sparse_weight_channels_dim = 0;
constexpr int dense_output_channels_dim = 1;

static std::vector<int64_t> sparse_conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[dense_input_batch_size_dim];
  output_size[1] = weight_size[sparse_weight_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

// Used on pad, stride and dilation
static void sparse_check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
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

static void sparse_convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {

  sparse_check_args(c, padding, input->dim() - 2, "padding");
  sparse_check_args(c, stride, padding.size(), "stride");
  sparse_check_args(c, dilation, padding.size(), "dilation");
  
  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, dense_input_channels_dim, weight->size(1) * groups);

  //Weight
  checkSameDim(c, input, weight);
  checkSameDim(c, input, output);
}

Tensor reshape_sparse_weight(const Tensor& weight_indices, uint32_t dim_2, uint32_t dim_3, uint64_t nnz) {

  TORCH_CHECK(weight_indices.size(0) == 4, "Indices should be a 4 x nnz matrix");
  IntTensor new_indices = at::zeros({nnz}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  uint32_t nnz_uint = (uint32_t)(nnz);
  hb_offload_kernel(new_indices, weight_indices, dim_2, dim_3, nnz_uint, "tensorlib_reshape_weight");
  return new_indices;
}

Tensor hb_sparse_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  //checkAllSameType(c, {input, weight});
  checkAllSameHB(c, {input, weight});
  
  auto output_t = at::zeros(
                    sparse_conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation, groups),
                    input->options());
  if (output_t.numel() == 0) {
    return output_t;
  }
    
  IntTensor indices = (*weight)._indices();
  Tensor values = (*weight)._values();
  IntArrayRef input_sizes = (*input).sizes();
  IntArrayRef weight_sizes = (*weight).sizes();
  uint64_t _nnz = (*weight)._nnz();
  uint32_t w_nnz = (uint32_t)(_nnz);
  uint64_t w_dim0 = (*weight).size(0);
  uint32_t w_dim1 = (uint32_t)((*weight).size(1));
  uint32_t w_dim2 = (uint32_t)((*weight).size(2));
  uint32_t w_dim3 = (uint32_t)((*weight).size(3));
  IntTensor new_indices = reshape_sparse_weight(indices, w_dim2, w_dim3, _nnz);
  
  IntTensor rowIndices = indices.select(0, 0);
  IntTensor csr = _to_csr_int(rowIndices, w_dim0, _nnz);

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };

  // sparse_convolution_shape_check(c, input, weight, output, padding, stride,
  //           dilation, groups);

  //Tensor weight_contig = weight->contiguous();
  hb_sparse_convolution_arg_check(output->dim(), dilation, groups);
  
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output_t, device_ptrs));
  device_args.push_back(create_device_tensor(*input, device_ptrs));
  device_args.push_back(create_device_tensor(csr, device_ptrs));
  device_args.push_back(create_device_tensor(new_indices, device_ptrs));
  device_args.push_back(create_device_tensor(values, device_ptrs));
  device_args.push_back(create_device_vector(padding, true, device_ptrs));
  device_args.push_back(create_device_vector(stride, true, device_ptrs));
  device_args.push_back(create_device_vector(input_sizes, true, device_ptrs));
  device_args.push_back(create_device_vector(weight_sizes, true, device_ptrs));
  
  c10::hammerblade::offload_kernel(
      "tensorlib_sparse_convolution_forward", device_args);
  cleanup_device(device_args, device_ptrs);
  
  return *output;
}

void hb_sparse_conv_add_bias_(CheckedFrom c, const TensorArg& output,
                              const TensorArg& bias) {
  checkAllSameType(c, {output, bias});
  checkAllSameHB(c, {output, bias});
  checkSize(c, bias, { output->size(dense_output_channels_dim) });

  if (output.tensor.numel() == 0) {
    return;
  }

  hb_offload_kernel(*output, *bias, "tensorlib_convolution_add_bias");
}

Tensor hb_sparse_convolution(
    const Tensor& input_t, const SparseTensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "hb_sparse_convolution";

  auto output_t = hb_sparse_convolution_forward(
    c, input, weight, padding, stride, dilation, groups);


  if (bias->defined()) {
    hb_sparse_conv_add_bias_(c, { output_t, "result", 0 }, bias);
  }

  return output_t;
}

}}
