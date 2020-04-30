#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

#include <iostream>

namespace at {
namespace native {

Tensor embedding_dense_backward_hb(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  TORCH_CHECK( !scale_grad_by_freq, "scale_grad_by_freq not yet supported on HB");

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);

  int64_t numel = indices.numel();
  auto indices_contig = indices.to(at::kInt).contiguous();
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int32_t padding_idx_i32 = safe_downcast<int32_t, int64_t>(padding_idx);
  int32_t num_weights_i32 = safe_downcast<int32_t, int64_t>(num_weights);
  int32_t output_numel_i32 = safe_downcast<int32_t, int64_t>(grad_.size(-1));

  hb_offload_kernel(grad_weight, grad, indices_contig, padding_idx_i32,
                    num_weights_i32, output_numel_i32,
                    "tensorlib_embedding_backward");

  return grad_weight;
}

}} // at::native
