#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>

namespace at { namespace native {

using DimMask = TensorIterator::DimMask;

Tensor& max_out_hb(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor _self, _other;
  std::tie(_self, _other) = expand_outplace(self, other, "max_out_hb");
  result.resize_as_(_self);
  auto iter = TensorIterator::binary_op(result, _self, _other,
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

template<typename scalar_type>
void offload_iterator_reduce_max_impl(TensorIterator& iter, Tensor& indices, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;

  auto N = (uint32_t) iter.numel();
  int64_t num_dim = iter.ndim();

   auto shape = iter.shape();

  int64_t* sizes = (int64_t*) malloc(iter.ndim() * sizeof(int64_t));
  TORCH_INTERNAL_ASSERT(sizes, "HB memory allocation failed on host");
  for(int i = 0; i < iter.ndim(); ++i) {
    sizes[i] = (int64_t) shape[i];
  }

  uint32_t size0 = sizes[0];

  for(uint32_t i=0; i<iter.ntensors(); ++i) {
    uint32_t n;

    auto iter_strides = iter.strides(i);
    int64_t* strides = (int64_t*) malloc(iter.ndim() * sizeof(int64_t));
    TORCH_INTERNAL_ASSERT(strides, "HB memory allocation failed on host");
    for(int j = 0; j < iter.ndim(); ++j) {
      strides[j] = (int64_t) (iter_strides[j] / sizeof(scalar_type));
    }
    
    if(i == 0) {
      n = iter.num_output_elements();
      sizes[0] = 1;
    } else {
      n = N;
      sizes[0] = size0;
    }

    eva_t device_arg = create_device_tensor(
        n, iter.ndim(),
        (const int64_t*)strides, (const int64_t*)sizes, iter.data_ptr(i),
#ifdef HB_ENABLE_KERNEL_LOG
        iter.tensor(i).storage().data(),
        (uint32_t) iter.tensor(i).storage().numel(),
#endif
        device_ptrs);
    device_args.push_back(device_arg);
 
    if(i == 0) {
      eva_t device_arg_1 = create_device_tensor(
        n, iter.ndim(),
        (const int64_t*)strides, (const int64_t*)sizes, indices.data_ptr(),
#ifdef HB_ENABLE_KERNEL_LOG
        indices.storage().data(),
        (uint32_t) indices.storage().numel(),
#endif
        device_ptrs);
      device_args.push_back(device_arg_1);
    }

    free(strides);
  }
  device_args.push_back(create_device_scalar((uint32_t)iter.num_reduce_dims()));

  free(sizes);

  c10::hammerblade::offload_kernel(kernel, device_args);
  cleanup_device(device_args, device_ptrs);

  iter.cast_outputs();
}
  

static TensorIterator make_max_reduction(const char* name, Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim, bool keepdim, ScalarType dtype) {
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
  return TensorIterator::reduce_op(result1, self);
}

static void  max_out_impl(Tensor& result, Tensor& indices, const Tensor& self, IntArrayRef dim, bool keepdim) {
  ScalarType dtype = self.scalar_type();
  auto iter = make_max_reduction("max_dim", result, indices, self, dim, keepdim, dtype);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "max_dim", [&]() {
      offload_iterator_reduce_max_impl<scalar_t>(iter, indices, "tensorlib_max_dim");
      });
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
