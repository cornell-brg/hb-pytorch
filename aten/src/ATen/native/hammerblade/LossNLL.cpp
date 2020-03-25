#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

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

template <typename scalar_t>
static void nll_loss_out_frame_hb(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  *total_weight_data = 0;

  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    output.resize_({batch_size});

    auto input_acc = input.accessor<scalar_t, 2>();
    auto target_acc = target.accessor<int64_t, 1>();
    auto output_acc = output.accessor<scalar_t, 1>();

    for (auto i = 0; i < batch_size; i++) {
      const auto cur_target = target_acc[i];

      if (cur_target == ignore_index) {
        output_acc[i] = 0;
        continue;
      }

      TORCH_CHECK_INDEX(
          cur_target >= 0 && cur_target < n_classes,
          "Target ",
          cur_target,
          " is out of bounds.");

      scalar_t cur_weight = weight_data != nullptr ? weight_data[cur_target]
                                                   : static_cast<scalar_t>(1);
      output_acc[i] = -input_acc[i][cur_target] * cur_weight;
    }

    return;
  }

  // produce scalar output when reducing or input is 1d
  output.resize_({});

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  const scalar_t* input_data = input_contiguous.data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.data_ptr<int64_t>();

  scalar_t output_val = 0;
  scalar_t total_weight_val = 0;

  if (input.dim() == 1) {
    const auto cur_target = target_data[0];
    if (cur_target != ignore_index) {
      TORCH_CHECK_INDEX(
          cur_target >= 0 && cur_target < n_classes,
          "Target ",
          cur_target,
          " is out of bounds.");
      total_weight_val =
          weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
      output_val = -input_data[cur_target] * total_weight_val;
    }
  } else if (input.dim() == 2) {
    const auto batch_size = input.size(0);
    TORCH_CHECK(target.size(0) == batch_size);
    const auto n_target = input.size(1);

    for (int64_t i = 0; i < batch_size; i++) {
      const auto cur_target = target_data[i];
      if (cur_target != ignore_index) {
        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        scalar_t cur_weight =
            weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
        total_weight_val += cur_weight;
        output_val -= input_data[i * n_target + cur_target] * cur_weight;
      }
    }
  }

  if (reduction == Reduction::Mean &&
      (total_weight_val != 0 || input.numel() == 0)) {
    // allow NaN result for total_weight_val == 0 case, see #15870
    output_val /= total_weight_val;
  }

  // write result to output tensors
  *output.data_ptr<scalar_t>() = output_val;
  *total_weight_data = total_weight_val;
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
        nll_loss_out_frame_hb<scalar_t>(
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
