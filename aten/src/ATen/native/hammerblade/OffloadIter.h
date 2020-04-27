#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>

namespace at {
namespace native {

//=======================================================================
// Offloading operations that use TensorIterator
//=======================================================================
void offload_iterator_op_impl(TensorIterator& iter, std::vector<eva_t> device_scalars,
    const char* kernel, uint32_t ntensors);

//=======================================================================
// Offloading operations that use reduction TensorIterator
//=======================================================================

template<typename scalar_type>
void offload_iterator_reduce_op_impl(TensorIterator& iter, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  TORCH_CHECK(iter.noutputs() == 1,
              "HB only support reduction with single output as of now");

  // Device pointers to tensors on the device
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;

  auto N = (uint32_t) iter.numel();

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
      // iterator strides are in bytes, so divide by data size
      strides[j] = (int64_t) (iter_strides[j] / sizeof(scalar_type));
    }

    if(i == 0) {
      // output tensor
      n = iter.num_output_elements();
      sizes[0] = 1;
    } else {
      n = N;
      sizes[0] = size0;
    }

    eva_t device_arg = create_device_tensor(
        n, iter.ndim(),
        (const int64_t*)strides, (const int64_t*)sizes, iter.data_ptr(i), device_ptrs);
    device_args.push_back(device_arg);
    free(strides);
  }
  device_args.push_back(create_device_scalar((uint32_t)iter.num_reduce_dims()));

  free(sizes);

  c10::hammerblade::offload_kernel(kernel, device_args);

  // Need to deallocate those args on device
  cleanup_device(device_args, device_ptrs);

  iter.cast_outputs();
}

//============================================
// TensorIterator operations with zero scalar
//============================================

void offload_op_nullary(TensorIterator& iter, const char* kernel);
void offload_op_unary(TensorIterator& iter, const char* kernel);
void offload_op_binary(TensorIterator& iter, const char* kernel);
void offload_op_ternary(TensorIterator& iter, const char* kernel);

//============================================
// TensorIterator operations with one scalar
//============================================

template <typename ST0>
void offload_op_nullary(TensorIterator& iter, ST0 alpha,
                        const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_nullary(sub_iter, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 1);
}

template <typename ST0>
void offload_op_unary(TensorIterator& iter, ST0 alpha,
                        const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_unary(sub_iter, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 2);
}

template <typename ST0>
void offload_op_binary(TensorIterator& iter, ST0 alpha,
                        const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_binary(sub_iter, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 3);
}

template <typename ST0>
void offload_op_ternary(TensorIterator& iter, ST0 alpha,
                        const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_ternary(sub_iter, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 4);
}

//============================================
// TensorIterator operations with two scalars
//============================================

template <typename ST0, typename ST1>
void offload_op_nullary(TensorIterator& iter, ST0 beta,
                        ST1 alpha, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_nullary(sub_iter, beta, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(beta));
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 1);
}

template <typename ST0, typename ST1>
void offload_op_unary(TensorIterator& iter, ST0 beta,
                        ST1 alpha, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_unary(sub_iter, beta, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(beta));
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 2);
}

template <typename ST0, typename ST1>
void offload_op_binary(TensorIterator& iter, ST0 beta,
                        ST1 alpha, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_binary(sub_iter, beta, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(beta));
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 3);
}

template <typename ST0, typename ST1>
void offload_op_ternary(TensorIterator& iter, ST0 beta,
                        ST1 alpha, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_ternary(sub_iter, beta, alpha, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(beta));
  scalars.push_back(create_device_scalar(alpha));

  offload_iterator_op_impl(iter, scalars, kernel, 4);
}


} // namespace native
} // namespace at
