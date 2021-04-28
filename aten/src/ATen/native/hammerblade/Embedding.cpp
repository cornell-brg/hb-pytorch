#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

#include <iostream>

namespace at {
namespace native {

<<<<<<< HEAD
Tensor embedding_dense_backward_hb(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
=======
// The idea here is to combine embedding with sum -- do reduction on the fly so we dont
// have to deal with strange memory layout in sum
Tensor embedding_hb(const Tensor & weight, const Tensor & indices,
                 int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding", indices_arg, kLong);
  TORCH_CHECK( indices.is_contiguous(), "assuming indices to be contiguous" );
  TORCH_CHECK( indices.ndimension() == 2, "currently requires a batch" );

  // std::cout << "in embedding_hb" << std::endl;
  // std::cout << "indices.sizes() = " << indices.sizes() << std::endl;
  // std::cout << "indices.strides() = " << indices.strides() << std::endl;

  int32_t padding_idx_i32 = safe_downcast<int32_t, int64_t>(padding_idx);
  auto sum = at::zeros({indices.size(0), weight.size(-1)}, weight.options());

  // std::cout << "sum.sizes() = " << sum.sizes() << " sum.strides() = " << sum.strides() << std::endl;

  hb_offload_kernel(sum, weight, indices, padding_idx_i32,
                    "tensorlib_embedding_with_sum");
  return sum.view({indices.size(0), 1, weight.size(-1)});
}

Tensor embedding_dense_backward_hb(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  TORCH_CHECK( !scale_grad_by_freq, "scale_grad_by_freq not yet supported on HB");

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);

  int64_t numel = indices.numel();
  auto indices_contig = indices.to(at::kInt).contiguous();
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)}).contiguous();
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int32_t padding_idx_i32 = safe_downcast<int32_t, int64_t>(padding_idx);
  int32_t num_weights_i32 = safe_downcast<int32_t, int64_t>(num_weights);
  int32_t output_numel_i32 = safe_downcast<int32_t, int64_t>(grad_.size(-1));

  hb_offload_kernel(grad_weight, grad, indices_contig, padding_idx_i32,
                    num_weights_i32, output_numel_i32,
                    "tensorlib_embedding_backward");

  return grad_weight;
}

Tensor embedding_dense_backward_hb_recsys(
    const Tensor & grad, const Tensor & indices, int64_t num_weights,
>>>>>>> d8d63e689c... [hack] now embedding hack of recsys is a stand alone op
    int64_t padding_idx, bool scale_grad_by_freq) {

  TORCH_CHECK( !scale_grad_by_freq, "scale_grad_by_freq not yet supported on HB");

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);

  int64_t numel = indices.numel();
  auto indices_contig = indices.to(at::kInt).contiguous();
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)}).contiguous();
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

  int32_t padding_idx_i32 = safe_downcast<int32_t, int64_t>(padding_idx);
  int32_t num_weights_i32 = safe_downcast<int32_t, int64_t>(num_weights);
  int32_t output_numel_i32 = safe_downcast<int32_t, int64_t>(grad_.size(-1));

  hb_offload_kernel(grad_weight, grad, indices_contig, padding_idx_i32,
                    num_weights_i32, output_numel_i32,
                    "tensorlib_embedding_backward_recsys");

  return grad_weight;
}

}} // at::native
