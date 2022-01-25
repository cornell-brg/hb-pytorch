#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

Tensor& min_out_hb(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor _self, _other;
  std::tie(_self, _other) = expand_outplace(self, other, "min_out_hb");
  result.resize_as_(_self);
  auto iter = TensorIterator::binary_op(result, _self, _other,
    /*check_mem_overlap=*/true);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "min", [&]() {
      offload_op_binary(iter, "tensorlib_min");
      });
  return result;
}

Tensor min_hb(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty(self.sizes(), self.options());
  return min_out_hb(result, self, other);
}

}}
