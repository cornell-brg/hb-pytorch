#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#ifdef USE_HB
#include <ATen/hammerblade/HammerBladeContext.h>
#endif

namespace at {
namespace native {

using namespace at::sparse;

SparseTensor llcopy_sparse_to_hb(const SparseTensor& self) {

#ifdef USE_HB
  // if tensor is not defined, just return it
  if (!self.defined()) {
    return self;
  }
  // get low level indices and values storage size
  Tensor cpu_indices = self._indices();
  Tensor values = self._values();
  IntTensor indices = cpu_indices.to(kInt);
  size_t indices_itemsize = indices.storage().itemsize();
  int64_t indices_numel = indices.storage().numel();
  size_t indices_storage_size = indices_numel * indices_itemsize;

  size_t values_itemsize = values.storage().itemsize();
  int64_t values_numel = values.storage().numel();
  size_t values_storage_size = values_numel * values_itemsize;

  // alloc on HB
  c10::Allocator* allocator = at::hammerblade::getHammerBladeDeviceAllocator();
  
  // device tensor reconstruction
  auto values_storage_offset = values.storage_offset();
  auto values_storage_impl = c10::make_intrusive<StorageImpl>(
      values.dtype(),
      values_numel,
      allocator->allocate(values_storage_size),
      allocator,
      /*resizeable=*/true);
  auto values_tensor = detail::make_tensor<TensorImpl>(std::move(values_storage_impl), at::TensorTypeId::HammerBladeTensorId);
  setStrided(values_tensor, values.sizes(), values.strides(), values_storage_offset);

  auto indices_storage_offset = indices.storage_offset();
  auto indices_storage_impl = c10::make_intrusive<StorageImpl>(
      indices.dtype(),
      indices_numel,
      allocator->allocate(indices_storage_size),
      allocator,
      /*resizeable=*/true);
  
  IntTensor indices_tensor = detail::make_tensor<TensorImpl>(std::move(indices_storage_impl), at::TensorTypeId::HammerBladeTensorId);
  setStrided(indices_tensor, indices.sizes(), indices.strides(), indices_storage_offset);

  // memcpy
  void* indices_ptr = (void*)indices.storage().data();
  void* hb_indices_ptr = (void*)indices_tensor.storage().data();
  c10::hammerblade::DMA_host_to_device(hb_indices_ptr, indices_ptr, indices_storage_size);

  void* values_ptr = (void*)indices.storage().data();
  void* hb_values_ptr = (void*)indices_tensor.storage().data();
  c10::hammerblade::DMA_host_to_device(hb_values_ptr, values_ptr, values_storage_size);

  //Create HB sparse tensor:
  SparseTensor sparse_tensor;
  get_sparse_impl(sparse_tensor)->set_indices_and_values_unsafe(indices_tensor, values_tensor);

  return sparse_tensor;

#else

  TORCH_CHECK(false, "cannot call llcopy_sparse_to_hb if not using HB");

#endif

}

}} // namespace at::native
