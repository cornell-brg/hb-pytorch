#include <ATen/ATen.h>
#include <ATen/native/hammerblade/Resize.h>
#include <ATen/native/ResizeCommon.h>

namespace at { 
namespace native {

Tensor& resize_hb_(
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

}} // namepsace at::native
