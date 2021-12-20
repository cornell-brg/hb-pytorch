#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {


Tensor& max_hb_self(const Tensor& self) {
    Tensor result = at::empty(self.sizes(), self.options());
    auto iter = TensorIterator::unary_op(result, self);
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "max_val_hb", [&]() {
      offload_op_unary(iter, "tensorlib_max_values");
      });
    return result;
}


// REGISTER_HAMMERBLADE_DISPATCH(max_values_stub, &max_hb_self);

}} // namespace at::native