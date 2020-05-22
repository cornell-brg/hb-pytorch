#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/ExpandUtils.h>

#include <iostream>

namespace at { namespace native {

Tensor vvadd_hb(
    const Tensor& self,
    const Tensor& other
) {

  if ((self.scalar_type() != ScalarType::Float) || (other.scalar_type() != ScalarType::Float)) {
    AT_ERROR("HammerBlade vvadd is implemented for Float only");
  }

  TORCH_CHECK(self.dim() == other.dim(), "Shapes of the two input tensors have to be the same");

  auto result = at::empty_like(self, self.options());
  auto iter = TensorIterator::binary_op(result, self, other, /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "vvadd_hb", [&]() {
      offload_op_binary(iter, "tensorlib_vvadd");
      });

  return result;
}

}} // namespace at::native
