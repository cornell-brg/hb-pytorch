#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

using DimMask = TensorIterator::DimMask;

Tensor& max_out_hb(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(result.sizes() == self.sizes() && result.sizes() == other.sizes(), "Tensor size should be equal");
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "max", [&]() {
      offload_op_binary(iter, "tensorlib_max");
      });
  return result;
}

Tensor max_hb(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty(self.sizes(), self.options());
  return max_out_hb(result, self, other);
}

static void make_max_reduction(const char* name, Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  TORCH_CHECK(!result1.defined() || result1.type().scalarType() == dtype,
        name, ": provided dtype must match dtype of result. Got ",
        toString(result1.type().scalarType()),
        " and ",
        toString(dtype),
        ".");
  TORCH_CHECK(!result2.defined() || result2.type().scalarType() == kInt, 
	name, ": the returened indices must be integer 32 dtype !");
  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result1, self, mask, keepdim, dtype);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  allocate_reduction_result(result2, self, mask, keepdim, kInt);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, false);
  uint32_t count = 0;
  for (int dim = 0; dim < ndim; dim++) {
    if(viewed_result1.stride(dim) == 0) {
      count++;
    }
  }
  hb_offload_kernel(result1, result2, self, count, "tensorlib_max_dim");
}

static void  max_out_impl(Tensor& result, Tensor& indices, const Tensor& self, IntArrayRef dim, bool keepdim) {
  ScalarType dtype = self.scalar_type();
  make_max_reduction("max_dim", result, indices, self, dim, keepdim, dtype);
}

Tensor maxall_hb(const Tensor& self) {
  Tensor result;
  Tensor indices;
  native::max_out_impl(result, indices, self, {}, false);
  return result;
}

std::tuple<Tensor, Tensor> max_dim_hb(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result;
  Tensor indices;
  native::max_out_impl(result, indices, self, dim, keepdim);
  return std::tuple<Tensor, Tensor>(result, indices);
}

std::tuple<Tensor&, Tensor&> max_dim_out_hb(Tensor& result, Tensor& indices, const Tensor& self, int64_t dim, bool keepdim) {
  native::max_out_impl(result, indices, self, dim, keepdim);
  return std::tuple<Tensor&, Tensor&>(result, indices);
}

}}
