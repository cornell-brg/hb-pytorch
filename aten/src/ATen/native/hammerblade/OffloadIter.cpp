#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>
#include <ATen/native/hammerblade/OffloadIter.h>

namespace at {
namespace native {


//=======================================================================
// Offloading operations that use TensorIterator
//=======================================================================

void offload_iterator_op_impl(TensorIterator& iter, std::vector<eva_t> device_scalars,
    const char* kernel, uint32_t ntensors) {

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_hammerblade());
  }

  // Device pointers to tensors on the device
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;

  auto N = (uint32_t)iter.numel();

  auto shape = iter.shape();

  int64_t* sizes = (int64_t*) malloc(iter.ndim() * sizeof(int64_t));
  TORCH_INTERNAL_ASSERT(sizes, "HB memory allocation failed on host");
  for(int i = 0; i < iter.ndim(); ++i) {
    sizes[i] = (int64_t) shape[i];
  }

  // buffer for changing strides layout
  int64_t *local_strides = (int64_t*) malloc(iter.ndim() * sizeof(int64_t));
  TORCH_INTERNAL_ASSERT(local_strides, "HB memory allocation failed on host");

  // Allocate device tensors and copy the data
  for(int i=0; i<iter.ntensors(); i++) {

    auto iter_strides = iter.strides(i);
    // ------------------------------
    // strides of iter are stored as:
    // dim0: tensor0, tensor1, tensor2
    // dim1: tensor0, tensor1, tensor2
    // ------------------------------
    // we want per tensor strides:
    // tensor0: dim0, dim1
    // ------------------------------
    for(int j=0; j<iter.ndim(); j++) {
      local_strides[j] = (int64_t)iter_strides[j];
    }
    // Iterate over all tensors to create
    // corresponding tensors on the device.
    eva_t device_arg = create_device_tensor(N, iter.ndim(),
        (const int64_t*)local_strides, (const int64_t*)sizes, iter.data_ptr(i), device_ptrs);
    device_args.push_back(device_arg);
  }

  // free strides buffer
  free(local_strides);
  free(sizes);

  // Add device scalars to arugments
  for(int i=0; i<device_scalars.size(); i++) {
    auto alpha = device_scalars[i];
    device_args.push_back(alpha);
  }

  c10::hammerblade::offload_kernel(kernel, device_args);

  // Need to deallocate those args on device
  cleanup_device(device_args, device_ptrs);

  iter.cast_outputs();
}


//=======================================================================
// Binary operations
//=======================================================================

void offload_op_binary(TensorIterator& iter, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_binary(sub_iter, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  offload_iterator_op_impl(iter, scalars, kernel, 3);
}


//=======================================================================
// Unary operations
//=======================================================================

void offload_op_unary(TensorIterator& iter, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_unary(sub_iter, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  offload_iterator_op_impl(iter, scalars, kernel, 2);
}

//=======================================================================
// Nullary operations
//=======================================================================

void offload_op_nullary(TensorIterator& iter, const char* kernel) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_nullary(sub_iter, kernel);
    }
    return;
  }

  std::vector<eva_t> scalars;
  offload_iterator_op_impl(iter, scalars, kernel, 1);
}

} // namespace native
} // namespace at
