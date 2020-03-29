#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/hammerblade/HammerBladeContext.h>

namespace at {
namespace native {

// Convert a single element Tensor to Scalar
Scalar _local_scalar_dense_hb(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "_local_scalar_dense_hb",
      [&] {
        TORCH_INTERNAL_ASSERT(self.device().is_hammerblade());
        scalar_t* value = (scalar_t*)malloc(sizeof(scalar_t));
        TORCH_CHECK(value, "Failed to allocate buffer for _local_scalar_dense_hb");
        void* dataptr = self.data_ptr();
        c10::hammerblade::memcpy_device_to_host((void*)value, dataptr, sizeof(scalar_t));
        r = Scalar(*value);
      });
  return r;
}

}} // at::native
