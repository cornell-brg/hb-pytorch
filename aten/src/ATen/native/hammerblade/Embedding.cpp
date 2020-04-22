#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor embedding_dense_backward_hb(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  TORCH_CHECK( !scale_grad_by_freq, "scale_grad_by_freq not yet supported on HB");

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);

  auto indices_contig = indices.contiguous().cpu();
  auto indices_data = indices_contig.data_ptr<int64_t>();
  int64_t numel = indices.numel();
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)});
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  // naive implementation
  for (size_t i=0; i<numel; i++) {
    if (indices_data[i] != padding_idx) {
      int64_t k = indices_data[i];
      if (k >= 0 && k < num_weights) {
        float scale = 1.0;
        grad_weight[k].add_(grad[i], scale);
      }
    }
  }

  return grad_weight;
}

}} // at::native
