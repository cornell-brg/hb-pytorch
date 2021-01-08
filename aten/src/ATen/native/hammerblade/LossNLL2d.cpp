#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {

namespace {

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// Returns the address of the first element of a tensor
// or nullptr if the tensor is undefined.
template <typename scalar_t>
inline scalar_t* optional_data(const Tensor& source) {
  return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
}

inline void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of dimension: ",
      target.dim());
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of dimension: ",
      input.dim());
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(
      input0 == target0 && input2 == target1 && input3 == target2,
      "size mismatch (got input: ",
      input.sizes(),
      " , target: ",
      target.sizes());
}

inline void check_gradout_shape_nll_loss2d(
    const Tensor& grad_output,
    const Tensor& target) {
  TORCH_CHECK(
      grad_output.dim() == 3,
      "grad_output must have same dimension as target (3) but got dimension: ",
      grad_output.sizes());

  const int64_t grad_output0 = grad_output.size(0);
  const int64_t grad_output1 = grad_output.size(1);
  const int64_t grad_output2 = grad_output.size(2);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(
      grad_output0 == target0 && grad_output1 == target1 &&
          grad_output2 == target2,
      "size mismatch (got grad_output: ",
      grad_output.sizes(),
      " target: ",
      target.sizes());
}

// Key core implementation

void nll_loss2d_forward_out_hb_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  if (reduction == Reduction::None) {
    // XXX: unimplemented
    TORCH_CHECK(false,"Reduction::None unimplemented");
    return;
  }

  // produce scalar outputs for the reduction case
  output.resize_({});

  auto input_contiguous  = input.contiguous();
  auto target_contiguous = target.contiguous();

  uint32_t reduction_u32    = safe_downcast<uint32_t, int64_t>(reduction);
  int32_t ignore_index_i32 = safe_downcast<int32_t, int64_t>(ignore_index);

  AT_DISPATCH_FLOAT_TYPE_ONLY(
      input.scalar_type(),
      "nll_loss2d_forward_out_frame",
      [&] {
        hb_offload_kernel(
            output,
            total_weight,
            input_contiguous,
            target_contiguous,
            reduction_u32,
            ignore_index_i32,
            "tensorlib_nll_loss2d_forward");
      });
}

void nll_loss2d_backward_out_hb_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  check_inputs_nll_loss2d(input, target, weight);
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  const auto target_contiguous = target.contiguous();

  TORCH_CHECK(
      grad_output.dim() <= 1 && grad_output.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      grad_output.sizes());

  AT_DISPATCH_FLOAT_TYPE_ONLY(
      input.scalar_type(),
      "nll_loss2d_backward_out_frame",
      [&] {
        hb_offload_kernel(
            grad_input,
            grad_output,
            input,
            target_contiguous,
            total_weight,
            reduction,
            ignore_index,
            "tensorlib_nll_loss2d_backward");
      });
}

} // namespace

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_hb(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  nll_loss2d_forward_out_hb_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_hb(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  nll_loss2d_forward_out_hb(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss2d_backward_out_hb(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  nll_loss2d_backward_out_hb_template(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_hb(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto grad_input = at::zeros_like(self);
  nll_loss2d_backward_out_hb(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

} // namespace native
} // namespace at
