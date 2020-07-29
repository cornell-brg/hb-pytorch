#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor _cat_hb(TensorList tensors, int64_t dim) {

  // TODO: more checks on these tensors are necessary??
  TORCH_CHECK(tensors.size() > 0,
              "_cat_hb: cannot concatenate empty tensor list");

  std::cout << "in _cat_hb, TensorList size = " << tensors.size()
            << " dim = " << dim << std::endl;

  // XXX: this is definitely wrong
  Tensor result = at::empty({42}, tensors[0].options());

  // convert TensorList length to uint32
  uint32_t length_u32 = safe_downcast<uint32_t, size_t>(tensors.size());
  // convert dim to int32
  int32_t dim_i32 = safe_downcast<int32_t, int64_t>(dim);

  // plain tensors
  std::vector<Tensor> tensor_args;
  tensor_args.push_back(result);

  // scalars
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(length_u32));
  scalars.push_back(create_device_scalar(dim_i32));

  // offload call
  offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");

  return result;

}

}} // namespace at::native
