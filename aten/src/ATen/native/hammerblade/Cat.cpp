//======================================================================
//   _cat_hb extension from Lin Cheng and Janice Wei
//   support cat on arbitrary dimension of 1-3D tensors
//   Zhongyuan Zhao 02/10/2021 zz546@cornell.edu
//======================================================================

#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor _cat_hb(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "_cat_hb: cannot concatenate empty tensor list");
  int64_t dims = tensors[0].dim();
  for(int i = 1; i < tensors.size(); i++) {
    TORCH_CHECK(tensors[i].dim() == dims, "The number of dimensions of input tensors should be equal");
  }
  TORCH_CHECK(dim < dims, "Invalid dimension parameter");
  TORCH_CHECK(tensors[0].dim() <= 3, "this simple cat only takes up to 3-dimension tensors");
  // convert TensorList length to uint32
  uint32_t length_u32 = safe_downcast<uint32_t, size_t>(tensors.size());
  // convert dim to int32
  int32_t dim_i32 = safe_downcast<int32_t, int64_t>(dim);

  // plain tensors
  std::vector<Tensor> tensor_args;

  // scalars
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(length_u32));
  scalars.push_back(create_device_scalar(dim_i32));

  //Compute the size of the output tensor
   
  
//  auto ndim = tensors[0].dim();
  uint32_t space = 0;
  for (size_t i = 0; i < length_u32; i++) {
    TORCH_CHECK(tensors[i].dim() == dims, "tensors have different dimensions");
    space += tensors[i].size(0);
  }

  Tensor result;
  if (dims == 1) {
    result = at::empty({space}, tensors[0].options());
  }
  else if (dims == 2) {
    result = at::empty({space, tensors[0].size(1)}, tensors[0].options());
  }
  else if (dims == 3) {
    result = at::empty({space, tensors[0].size(1), tensors[0].size(2)}, tensors[0].options());
  }

  tensor_args.push_back(result);
  // offload call
  offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");
  return result;
}

}} // namespace at::native
