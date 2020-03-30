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

//============================================
// TensorIterator operations with zero scalar
//============================================

void offload_op_nullary(TensorIterator& iter, const char* kernel);
void offload_op_unary(TensorIterator& iter, const char* kernel);
void offload_op_binary(TensorIterator& iter, const char* kernel);

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


} // namespace native
} // namespace at
