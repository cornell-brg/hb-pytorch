#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& remainder_out_hb(Tensor& result, const Tensor& self, const Tensor& other) {
  
//  TORCH_CHECK(result.sizes() == self.sizes() && result.sizes() == other.sizes(), "Tensor size should be equal");
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  AT_DISPATCH_INT_TYPE_ONLY(iter.dtype(), "remainder", [&]() {
      offload_op_binary(iter, "tensorlib_remainder");
      });
  return result;
}

Tensor remainder_hb(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty(self.sizes(), self.options());
  return remainder_out_hb(result, self, other);
}

}}
