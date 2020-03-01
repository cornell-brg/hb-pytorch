#pragma once

#include <ATen/ATen.h>
#include <TH/THTensor.hpp>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

static inline void HBStorage_resize(THStorage* storage, ptrdiff_t size) {
  if (storage->resizable()) {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr new_data;
    if (size != 0) {
      new_data = storage->allocator()->allocate(storage->itemsize() * size);
    }
    at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
    ptrdiff_t old_size = storage->numel();
    storage->set_numel(size);
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if (storage->numel() < copy_size) {
        copy_size = storage->numel();
      }
      if (copy_size > 0) {
        offload_memcpy(
            (eva_t) ((intptr_t) storage->data()),
            (eva_t) ((intptr_t) old_data.get()),
            storage->itemsize() * copy_size);
      }
    }
  } else {
    AT_ERROR("Trying to resize storage that is not resizable");
  }
}

static inline void maybe_resize_storage_hb(TensorImpl* self, int64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.cuh)
  if (new_size > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      HBStorage_resize(
          THTensor_getStoragePtr(self),
          new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_hb_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_hb(self, storage_size);

  return self;
}

}} // namepsace at::native
