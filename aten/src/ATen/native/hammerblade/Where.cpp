#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor _s_where_hb(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
  Tensor result = at::empty(self.sizes(), self.options());

  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(condition);
  iter.add_input(self);
  iter.add_input(other);
  iter.dont_compute_common_dtype();
  iter.build();

  if (condition.scalar_type() == at::ScalarType::Byte) {
    AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "where_byte_hb", [&](){
      offload_op_ternary(iter, "tensorlib_where_byte");
    });
  } else {
    AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "where_bool_hb", [&](){
      offload_op_ternary(iter, "tensorlib_where_bool");
    });
  }

  return result;
}

}}
