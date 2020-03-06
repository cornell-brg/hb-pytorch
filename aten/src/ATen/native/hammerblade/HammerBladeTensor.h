#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>

namespace at {
namespace native {

inline c10::StorageImpl* hammerblade_getStoragePtr(const c10::TensorImpl &tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  TORCH_CHECK(tensor.storage(), "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor.storage().unsafeGetStorageImpl();
}

} // namespace native
} // namespace at
