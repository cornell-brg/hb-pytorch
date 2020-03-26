#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

namespace {

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

inline Tensor optional_to_float(const Tensor& source) {
  return source.defined() ? source.to(at::kFloat) : source;
}

//--------------------------------------------------------
// Forward pass
//--------------------------------------------------------

static void nll_loss_out_frame_hb(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {

  const auto n_dims = input.dim();
  auto input_contiguous = input.contiguous();
  // convert to int32, and contiguous tensors
  auto target_contiguous = target.to(at::kInt).contiguous();
  auto weight_contiguous = optional_contiguous(weight);
  auto weight_float = optional_to_float(weight_contiguous);

  if (n_dims == 1 || reduction != Reduction::None) {
    // produce scalar output when reducing or input is 1d
    output.resize_({});
  } else {
    const auto batch_size = input.size(0);
    output.resize_({batch_size});
  }

  uint32_t reduction_u32 = safe_downcast<uint32_t, int64_t>(reduction);
  int32_t ignore_index_i32 = safe_downcast<int32_t, int64_t>(ignore_index);

  if (weight_float.defined()) {
    hb_offload_kernel(output, total_weight, input_contiguous,
                      target_contiguous, weight_float,
                      reduction_u32, ignore_index_i32,
                      "tensorlib_lossnll_weight");
  } else {
    hb_offload_kernel(output, total_weight, input_contiguous,
                      target_contiguous,
                      reduction_u32, ignore_index_i32,
                      "tensorlib_lossnll");
  }

}

void nll_loss_forward_out_hb_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  const auto n_classes = input.size(-1);

  TORCH_CHECK(
      !weight.defined() || weight.numel() == n_classes,
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  total_weight.resize_({});

  AT_DISPATCH_FLOAT_TYPE_ONLY(input.scalar_type(), "nll_loss_out_frame_hb",
      [&] {
        nll_loss_out_frame_hb(
            output,
            total_weight,
            input,
            target,
            weight,
            reduction,
            ignore_index);
      });
}


//--------------------------------------------------------
// Backward pass
//--------------------------------------------------------

static void nll_loss_backward_out_frame_hb(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {

  const auto n_dims = input.dim();
  auto grad_output_contiguous = grad_output.contiguous();
  auto input_contiguous = input.contiguous();
  // convert to int32, and contiguous tensors
  auto target_contiguous = target.to(at::kInt).contiguous();
  auto weight_contiguous = optional_contiguous(weight);
  auto weight_float = optional_to_float(weight_contiguous);
  // total_weight should be 0-dim

  uint32_t reduction_u32 = safe_downcast<uint32_t, int64_t>(reduction);
  int32_t ignore_index_i32 = safe_downcast<int32_t, int64_t>(ignore_index);

  if (weight_float.defined()) {
    hb_offload_kernel(grad_input, grad_output_contiguous,
                      input_contiguous, target_contiguous,
                      weight_float, total_weight,
                      reduction_u32, ignore_index_i32,
                      "tensorlib_lossnll_backward_weight");
  } else {
    hb_offload_kernel(grad_input, grad_output_contiguous,
                      input_contiguous, target_contiguous,
                      total_weight,
                      reduction_u32, ignore_index_i32,
                      "tensorlib_lossnll_backward");
  }

}

void nll_loss_backward_out_hb_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  grad_input.resize_as_(input);
  grad_input.zero_();

  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(-1),
      "weight tensor should be defined either for all or no classes");

  AT_DISPATCH_FLOAT_TYPE_ONLY(input.scalar_type(),
      "nll_loss_backward_out_frame_hb",
      [&] {
        nll_loss_backward_out_frame_hb(
            grad_input,
            grad_output,
            input,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight);
      });
}

} // namespace


//--------------------------------------------------------
// Forward pass
//--------------------------------------------------------

std::tuple<Tensor, Tensor> nll_loss_forward_hb(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  nll_loss_forward_out_hb(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

std::tuple<Tensor&, Tensor&> nll_loss_forward_out_hb(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  nll_loss_forward_out_hb_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

//--------------------------------------------------------
// Backward pass
//--------------------------------------------------------

Tensor& nll_loss_backward_out_hb(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  nll_loss_backward_out_hb_template(
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

Tensor nll_loss_backward_hb(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  nll_loss_backward_out_hb(
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
