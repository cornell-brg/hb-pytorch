#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& max_out_hb(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(result.sizes() == self.sizes() && result.sizes() == other.sizes(), "Tensor size should be equal");
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "max", [&]() {
      offload_op_binary(iter, "tensorlib_max");
      });
  return result;
}

Tensor max_hb(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty(self.sizes(), self.options());
  return max_out_hb(result, self, other);
}

Tensor max_hb(const Tensor& self) {
  Tensor result = at::empty(self.sizes(), self.options());
  auto iter = TensorIterator::unary_op(result, self);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "max_val_hb", [&]() {
    offload_op_unary(iter, "tensorlib_max_values");
  });
  return result;
}

}}
