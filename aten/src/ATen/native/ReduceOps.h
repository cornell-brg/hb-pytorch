#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);
DECLARE_DISPATCH(reduce_fn, and_stub);
DECLARE_DISPATCH(reduce_fn, or_stub);
DECLARE_DISPATCH(reduce_fn, min_values_stub);
DECLARE_DISPATCH(reduce_fn, max_values_stub);
DECLARE_DISPATCH(reduce_fn, argmax_stub);
DECLARE_DISPATCH(reduce_fn, argmin_stub);

using reduce_std_var_function =
  void (*)(TensorIterator&, bool unbiased, bool take_sqrt);
DECLARE_DISPATCH(reduce_std_var_function, std_var_stub);

using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, Scalar, c10::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

using reduce_fn_flag = void(*)(TensorIterator &, Scalar);
DECLARE_DISPATCH(reduce_fn_flag, norm_stub);

at::TensorIterator::DimMask make_dim_mask(IntArrayRef dims, int64_t ndim);
void allocate_reduction_result(Tensor& result, const Tensor& self, at::TensorIterator::DimMask mask, bool keepdim, ScalarType dtype);
Tensor review_reduce_result(const Tensor& result, int ndim, at::TensorIterator::DimMask mask, bool keepdim);

}} // namespace at::native
