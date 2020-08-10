#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Resize.h>
#include <ATen/native/ResizeCommon.h>

namespace at { namespace native {

Tensor& resize_h_(
    Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
#ifdef BUILD_NAMEDTENSOR
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
#endif
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_hb_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

Tensor _cat_hb(TensorList tensors, int64_t dim) {

  std::cout << "in _cat_hb, TensorList size = " << tensors.size()
            << " dim = " << dim << std::endl;
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

  // XXX: this is definitely wrong
  Tensor result = at::empty({0}, tensors[0].options());
  bool allSkipped = true;
  bool allContiguous = true;
  Tensor notSkippedTensor;

  auto should_skip = [](const Tensor& t) { return t.numel() == 0 && t.dim() == 1; };
  for (auto const &tensor : tensors) {
    if (should_skip(tensor)) {
      continue;
    }
    // we've found a non-empty tensor
    allSkipped = false;
    notSkippedTensor = tensor;
    break;
  }
  if (allSkipped) {
	tensor_args.push_back(result);
	// offload call
    offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");
    return result;
  }

  TORCH_CHECK(tensors.size() > 0, "expected a non-empty list of Tensors");
  TORCH_CHECK(dim <= notSkippedTensor.dim(), "dimension ", dim, "out of range");

  // when the input tensors are of the same size and strides,
  // reuse the same iterator for all input tensors
  bool reuse_iterator = true;
  bool no_type_promotion = true;
  // Check the type of the result
  no_type_promotion = result.dtype() == notSkippedTensor.dtype();

  // compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  auto first_tensor_mem_format = tensors[0].suggest_memory_format();
  for (int i = 0; i < tensors.size(); i++) {
    auto const &tensor = tensors[i];
    if (should_skip(tensor)) {
      // don't use fast path for empty tensor
      allContiguous = false;
      continue;
    }

    cat_dim_size += tensor.size(dim);

    if (!tensor.is_contiguous(first_tensor_mem_format)) {
      allContiguous = false;
    }

    if (tensor.sizes() != notSkippedTensor.sizes() ||
        tensor.strides() != notSkippedTensor.strides()) {
      reuse_iterator = false;
    }
    if (tensor.dtype() != notSkippedTensor.dtype()) {
      no_type_promotion = false;
    }
  }
  // compute the size of the result
  auto result_size = notSkippedTensor.sizes().vec();
  result_size[dim] = cat_dim_size;
  result = resize_h_(result, result_size, first_tensor_mem_format);
  if (result.numel() == 0) {
	tensor_args.push_back(result);
	// offload call
    offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");
    return result;
  }

  // fast path for single thread when both inputs and result are contiguous and not empty
  allContiguous = allContiguous && result.is_contiguous(first_tensor_mem_format);
  bool use_serial_kernel = result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
  ScalarType dtype = notSkippedTensor.scalar_type();
  if (use_serial_kernel && allContiguous && no_type_promotion && (dtype == ScalarType::Double || dtype == ScalarType::Float)) {
  tensor_args.push_back(result);
  // offload call
  offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");

  return result;
  }

  int32_t offset = 0;
  if (reuse_iterator &&
      result.is_contiguous(first_tensor_mem_format) &&
      no_type_promotion) {
    auto source_slice = notSkippedTensor;
    auto slice_dim_size = source_slice.size(dim);
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());

    auto iter = TensorIterator();
    iter.add_output(result_slice);
    iter.add_input(source_slice);
    iter.build();

    for (auto const &tensor : tensors) {
      if (should_skip(tensor)) {
        continue;
      }
      auto source_data = static_cast<char*>(tensor.data_ptr());
      auto result_data = static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, source_data);
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (auto const &tensor: tensors) {
      if (should_skip(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.size(dim);
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = TensorIterator();
      iter.add_output(result_slice);
      iter.add_input(tensor);
      iter.build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }
  tensor_args.push_back(result);
  // offload call
  offload_tensorlist_scalar_impl(tensors, tensor_args, scalars, "tensorlib__cat");

  return result;


}

}} // namespace at::native
