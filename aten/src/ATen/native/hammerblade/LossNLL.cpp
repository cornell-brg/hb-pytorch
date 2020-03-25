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


static void nll_loss_out_frame_hb(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {

  const auto n_dims = input.dim();
  // convert to int32, and contiguous tensors
  auto weight_contiguous = optional_contiguous(weight);
  auto weight_float = optional_to_float(weight_contiguous);
  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.to(at::kInt).contiguous();

  if (n_dims == 1 || reduction != Reduction::None) {
    // produce scalar output when reducing or input is 1d
    output.resize_({});
  } else {
    const auto batch_size = input.size(0);
    output.resize_({batch_size});
  }

  uint32_t reduction_u32 = safe_downcast<uint32_t, int64_t>(reduction);
  uint32_t ignore_index_u32 = safe_downcast<uint32_t, int64_t>(ignore_index_u32);

  if (weight_float.defined()) {
    hb_offload_kernel(output, total_weight, input_contiguous,
                      target_contiguous, weight_float,
                      reduction_u32, ignore_index_u32,
                      "tensorlib_lossnll_weight");
  } else {
    hb_offload_kernel(output, total_weight, input_contiguous,
                      target_contiguous,
                      reduction_u32, ignore_index_u32,
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

} // namespace

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

} // namespace native
} // namespace at
